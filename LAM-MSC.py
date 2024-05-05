import SCwithCGE
import MMA
import LKB

if __name__ == '__main__':
    # an example for transmitting images
    img_path = "imgs" # your image path
    # Modal transformation based on MMA
    texts = MMA.img2text(img_path)
    # Semantic Extraction Based on LKB
    userInfo = {"name": "Mike", "interests": "running", "language": "English", "identify": "student", "gender": "male"}
    personalized_semantics = []
    for input_text in texts:
        personalized_semantic = LKB.personalized_semantics(userInfo, input_text)
        personalized_semantics.append(personalized_semantic)
    # Data Transmission Based on CGE Assisted-SC
    rec_texts = SCwithCGE.data_transmission(personalized_semantics)
    # Semantic Recovery Based on LKB
    userInfo = {"name": "Jane", "interests": "shopping", "language": "English", "identify": "student","gender":"female"}
    personalized_semantics = []
    for input_text in rec_texts:
        personalized_semantic = LKB.personalized_semantics(userInfo, input_text)
        personalized_semantics.append(personalized_semantic)
    # Modal recovery based on MMA
    MMA.text2img(personalized_semantics)