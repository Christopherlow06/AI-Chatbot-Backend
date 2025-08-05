Questions=[
    "When looking onto the output shaft, what is the standard direction of rotation?",
    "What is an example case to contact SEW-EURODRIVE?",
    "What is the description for right-angle gear unit with the designation 'WHF..'?",
    "What is the View of output at end B of a 'W' Series gear unit at stage 3?"
    "What is the Flange diameter of the gear unit SF67p?"
    "What is the tightening torque in Nm for gear unit, SF87p with flange diameter of 350 and using screw/nut M16?",
    "What are the steps to mount a shaft-mounted gear unit with splined hollow shaft?",
    "What are the steps to activating the breather valve?",
    "What are the steps to mounting the cover?",
]


Pages=[
    r"C:\Users\sgdrig01\Desktop\AI App Internship project\Testing Data\SPRIOPLAN_gearmotors_pics\page_33.png",
    r"C:\Users\sgdrig01\Desktop\AI App Internship project\Testing Data\SPRIOPLAN_gearmotors_pics\page_36.png",
    r"C:\Users\sgdrig01\Desktop\AI App Internship project\Testing Data\SPRIOPLAN_gearmotors_pics\page_29.png",
    r"C:\Users\sgdrig01\Desktop\AI App Internship project\Testing Data\SPRIOPLAN_gearmotors_pics\page_34.png",
    r"C:\Users\sgdrig01\Desktop\AI App Internship project\Testing Data\SPRIOPLAN_gearmotors_pics\page_37.png",
    r"C:\Users\sgdrig01\Desktop\AI App Internship project\Testing Data\SPRIOPLAN_gearmotors_pics\page_39.png",
    r"C:\Users\sgdrig01\Desktop\AI App Internship project\Testing Data\SPRIOPLAN_gearmotors_pics\page_55.png",
    r"C:\Users\sgdrig01\Desktop\AI App Internship project\Testing Data\SPRIOPLAN_gearmotors_pics\page_45.png",
    r"C:\Users\sgdrig01\Desktop\AI App Internship project\Testing Data\SPRIOPLAN_gearmotors_pics\page_82.png",
]


Parameters=[
    {
        "num_ctx": 512,
        "temperature": 1.35,
        "top_k":50,
        "repeat_penalty": 1.0,
        "mirostat_mode": 2.0,
        "mirostat_tau": 3.0,
        "seed": 560
    },
    {
        "num_ctx": 512,
        "temperature": 1.3,
        "top_k":50,
        "repeat_penalty": 1.0,
        "mirostat_mode": 2.0,
        "mirostat_tau": 3.0,
        "seed": 561
    },
    {
        "num_ctx": 512,
        "temperature": 1.25,
        "top_k":50,
        "repeat_penalty": 1.0,
        "mirostat_mode": 2.0,
        "mirostat_tau": 3.0,
        "seed": 562
    },
]


input={
    "Question": Questions,
    "Page": Pages,
}

################################################################################################
question=[]
answer1=[]
time_taken1=[]
answer2=[]
time_taken2=[]
answer3=[]
time_taken3=[]

output={
    "question": question,
    "answer1": answer1,
    "time_taken1": time_taken1,
    "answer2": answer2,
    "time_taken2": time_taken2,
    "answer3": answer3,
    "time_taken3": time_taken3,
}