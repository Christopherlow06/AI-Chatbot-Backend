Parameters=[
    #lowering temperature
    {
        "num_ctx": 512,
        "temperature": 1.4,
        "top_k":50,
        "repeat_penalty": 1.0,
        "mirostat_mode": 2.0,
        "mirostat_tau": 3.0,
        "seed": 520
    },
    {
        "num_ctx": 512,
        "temperature": 1.3,
        "top_k":50,
        "repeat_penalty": 1.0,
        "mirostat_mode": 2.0,
        "mirostat_tau": 3.0,
        "seed": 521
    },
    {
        "num_ctx": 512,
        "temperature": 1.2,
        "top_k":50,
        "repeat_penalty": 1.0,
        "mirostat_mode": 2.0,
        "mirostat_tau": 3.0,
        "seed": 522
    },
    {
        "num_ctx": 512,
        "temperature": 1.1,
        "top_k":50,
        "repeat_penalty": 1.0,
        "mirostat_mode": 2.0,
        "mirostat_tau": 3.0,
        "seed": 523
    },
    {
        "num_ctx": 512,
        "temperature": 1.0,
        "top_k":50,
        "repeat_penalty": 1.0,
        "mirostat_mode": 2.0,
        "mirostat_tau": 3.0,
        "seed": 524
    },
    # lowering top_k
    {
        "num_ctx": 512,
        "temperature": 1.5,
        "top_k":40,
        "repeat_penalty": 1.0,
        "mirostat_mode": 2.0,
        "mirostat_tau": 3.0,
        "seed": 525
    },
    {
        "num_ctx": 512,
        "temperature": 1.5,
        "top_k":30,
        "repeat_penalty": 1.0,
        "mirostat_mode": 2.0,
        "mirostat_tau": 3.0,
        "seed": 526
    },
    {
        "num_ctx": 512,
        "temperature": 1.5,
        "top_k":20,
        "repeat_penalty": 1.0,
        "mirostat_mode": 2.0,
        "mirostat_tau": 3.0,
        "seed": 527
    },
    {
        "num_ctx": 512,
        "temperature": 1.5,
        "top_k":10,
        "repeat_penalty": 1.0,
        "mirostat_mode": 2.0,
        "mirostat_tau": 3.0,
        "seed": 528
    },
]