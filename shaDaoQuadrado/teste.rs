use openssl::sha::sha256;

static hash = sha256(b"your data or message");
println!("Hash = {}", hex::encode(hash));


