import os
from ocr_engine import init_ocr_reader, extract_text_from_image
from models import extract_entities_from_text
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from engines.credibility_engine import Run_Objective_2_Filter
import subprocess
import time
import atexit
import socket

# Keep track of processes to terminate them later
_running_processes = []

def cleanup_processes():
    for p in _running_processes:
        try:
            p.terminate()
        except:
            pass
    print("\n[Cleanup] Stopped background services.")

atexit.register(cleanup_processes)

def wait_for_port(port, host='localhost', timeout=10.0):
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            if result == 0:
                return True
        time.sleep(0.5)
    return False

def start_background_services():
    print("\n--- Starting Objective 3 Background Services ---")
    
    # 1. Start WhisperLive Server (Port 9090)
    print("  Starting WhisperLive Server (whisper_Test/server.py)...")
    p_whisper = subprocess.Popen(["python", "server.py"], cwd="whisper_Test", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _running_processes.append(p_whisper)
    if not wait_for_port(9090, timeout=15):
        print("  ⚠️ WhisperLive server might not have started correctly on port 9090.")
    else:
        print("  ✅ WhisperLive server running on port 9090.")

    # 2. Start WebRTC Server (Port 3000)
    print("  Starting WebRTC Node Server (webrtc-call/server)...")
    p_webrtc = subprocess.Popen(["node", "server.js"], cwd="webrtc-call/server", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _running_processes.append(p_webrtc)
    if not wait_for_port(3000):
        print("  ⚠️ WebRTC server might not have started correctly on port 3000.")
    else:
        print("  ✅ WebRTC server running on port 3000.")

    # 3. Start Call Bridge (Port 5000)
    print("  Starting Call Bridge (whisper_Test/call_bridge.py)...")
    p_bridge = subprocess.Popen(["python", "call_bridge.py"], cwd="whisper_Test", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _running_processes.append(p_bridge)
    if not wait_for_port(5000):
        print("  ⚠️ Call bridge might not have started correctly on port 5000.")
    else:
        print("  ✅ Call bridge running on port 5000.")
        
    print("  All required Objective 3 services started.")

def run_image_test(image_path: str):
    print("=" * 60)
    print("📸 STARTING END-TO-END OCR + OBJECTIVE 1, 2 & 3 PIPELINE TEST")
    print("=" * 60)

    # 1. INITIALIZE EASYOCR & EXTRACT TEXT FROM IMAGE
    print("\n--- Step 1: Extracting Raw Text via EasyOCR ---")
    reader = init_ocr_reader(lang_list=['en'], gpu=False)
    
    raw_ocr_text = extract_text_from_image(reader, image_path)
    
    if not raw_ocr_text:
        print("❌ Error: Could not extract text from image.")
        return

    print(f"✅ Extracted {len(raw_ocr_text)} characters of raw text from image.")
    print("--- RAW OCR PREVIEW ---")
    print(raw_ocr_text[:300] + "...\n----------------------")

    # 2. OBJECTIVE 1a: ENTITY EXTRACTION VIA LOCAL OLLAMA
    print("\n--- Step 2: Extracting Structured Entities with Ollama ---")
    entities_obj = extract_entities_from_text(raw_ocr_text)
    
    entities = {
        "company_name": entities_obj.company_name,
        "job_role": entities_obj.job_role,
        "recruiter_name": entities_obj.recruiter_name,
        "recruiter_email": entities_obj.recruiter_email,
        "domain_or_website": entities_obj.domain_or_website,
        "location": entities_obj.location
    }
    
    print("Parsed Entities:")
    print(f"  • Company : {entities['company_name']}")
    print(f"  • Role    : {entities['job_role']}")
    print(f"  • Email   : {entities['recruiter_email']}")
    print(f"  • Domain  : {entities['domain_or_website']}")
    print(f"  • Location: {entities['location']}")

    # 3. OBJECTIVE 1b: FAISS DATABASE SEARCH
    print("\n--- Step 3: Querying FAISS Vector Database ---")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    
    query = f"{entities['company_name']} {entities['job_role']}".strip()
    results = vector_db.similarity_search_with_score(query, k=3)
    
    retrieved_evidence = []
    for doc, score in results:
        retrieved_evidence.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "distance": float(score)
        })
    print(f"Retrieved {len(retrieved_evidence)} evidence documents from FAISS.")

    # 4. OBJECTIVE 2: CREDIBILITY FILTRATION LAYER
    print("\n--- Step 4: Running Credibility Filtration Layer ---")
    results_obj2 = Run_Objective_2_Filter(entities, retrieved_evidence)
    
    print("\n================ OBJECTIVE 2 REPORT ================")
    print(f"TRUST SCORE : {results_obj2['final_trust_score']}/100")
    print(f"RISK LEVEL  : {results_obj2['risk_level']}")
    print(f"FLAGS       : {results_obj2['flags_raised']}")
    print(f"\nM-RAV ANALYSIS:\n{results_obj2['mrav_explanation']}")
    print("====================================================")

    # 5. OBJECTIVE 3: VOICE VERIFICATION BRIDGE (conditional)
    #    This step only runs if Obj2 flags the offer as suspicious
    #    and routes to voice verification.
    print("\n--- Step 5: Voice Verification Bridge (Obj2 → Obj3 → DST) ---")
    try:
        from voice_verification_bridge import run_full_verification
        from schemas.credibility_schema import Objective2OutputPayload

        # Determine if voice verification is warranted based on Obj2 results
        # Use the risk level from Obj2's simple filter as the trigger
        should_verify = results_obj2['risk_level'] in ("HIGH RISK", "MEDIUM RISK")

        if should_verify:
            print(f"  ⚠️  Risk level is {results_obj2['risk_level']} — voice verification recommended.")
            print("  📞 Note: Full voice verification requires the WebRTC bridge and")
            print("     Node.js server to be running. The pipeline will now start these")
            print("     automatically for you.\n")
            
            start_background_services()

            # Build a minimal Obj2 output for the bridge
            # The bridge accepts the formal Objective2OutputPayload, but we can
            # also demonstrate the DST combination with just the simple filter output
            from obj2dst import DSTEngine

            dst = DSTEngine()
            print("  Building DS masses from retrieved evidence...")
            masses = []
            for ev in retrieved_evidence:
                is_fraud = ev.get("metadata", {}).get("fraudulent", 0)
                sim = max(0.0, 1.0 - ev.get("distance", 1.0))  # Convert distance to similarity
                mass = dst.assign_mass_function(
                    similarity_score=sim,
                    source_type=ev.get("metadata", {}).get("source_type", "unknown"),
                    is_fraudulent=is_fraud if is_fraud is not None else 0,
                )
                masses.append(mass)
                print(f"    📄 {ev.get('metadata', {}).get('source_type', '?')}: "
                      f"REAL={mass['REAL']:.3f}, FAKE={mass['FAKE']:.3f}")

            if masses:
                combined = dst.combine_all_evidence(masses)
                belief = dst.get_belief_strength(combined)
                conflict = dst.get_conflict_info(combined)

                print(f"\n  📊 DST Combined: REAL={combined['REAL']:.4f}, FAKE={combined['FAKE']:.4f}")
                print(f"  🏷️  Verdict: {belief}")
                print(f"  ⚖️  {conflict}")
        else:
            print(f"  ✅ Risk level is {results_obj2['risk_level']} — voice verification not needed.")
            print("  Proceeding directly to confidence framework.")

    except ImportError as e:
        print(f"  ⚠️  Bridge module not available ({e}). Skipping voice verification step.")
    except Exception as e:
        print(f"  ❌ Voice bridge error: {e}")

    # ── Final Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📋 FULL PIPELINE SUMMARY (Obj1 → Obj2 → Obj3)")
    print("=" * 60)
    print(f"  Image:        {image_path}")
    print(f"  Company:      {entities['company_name']}")
    print(f"  Role:         {entities['job_role']}")
    print(f"  Obj2 Trust:   {results_obj2['final_trust_score']}/100")
    print(f"  Obj2 Risk:    {results_obj2['risk_level']}")
    print(f"  Obj2 Flags:   {len(results_obj2['flags_raised'])} flag(s)")
    print("=" * 60)

if __name__ == "__main__":
    # Make sure richmond_offer.png exists in your FakeJobs folder
    IMAGE_FILE = "richmond_offer.png" 
    
    if os.path.exists(IMAGE_FILE):
        run_image_test(IMAGE_FILE)
    else:
        print(f"❌ File '{IMAGE_FILE}' not found. Please place the offer letter image in your folder!")