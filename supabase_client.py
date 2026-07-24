import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
url:str = os.environ.get("SUPABASE_URL")
key:str = os.environ.get("SUPABASE_KEY")
supabase:Client = create_client(url,key)

def get_all_users():
    response = (
        supabase.table("profiles")
        .select("id")
        .execute()
    )

    return response

def get_all_pets():
    response = (
        supabase.table("pets")
        .select("*")
        .execute()
    )

    return response

async def insert_new_pet(embedding, image_bytes: bytes, file_extension: str, name: str):
    pet_name = name
    pet_type_id = get_pet_type_id("cat")
    pet_status_id = get_pet_status_id("registered")
    owner_id = get_owner_id("markwell@gmail.com")
    municipality_id = get_municipality_id()
    pet_avatar_url = upload_pet_avatar(pet_name, image_bytes, file_extension).path

    response = (
        supabase.table("pets")
        .insert({"name": pet_name, "type": pet_type_id, "status": pet_status_id, "owner": owner_id, "avatar_url": pet_avatar_url, "place_of_registration": municipality_id, "embedding": embedding})
        .execute()
    )

    return response

def upload_pet_avatar(user_id:str, image: bytes, file_extension: str):
    response = (
        supabase.storage
        .from_("pet_avatars")
        .upload(
            file=image,
            path=f"{user_id}.{file_extension}",
            file_options={"cache-control": "3600", "upsert": "false", "content-type": f"image/{file_extension}"}
        )
    )

    return response


def get_pet_type_id(pet_type:str):
    response = (
        supabase.table("pet_type")
        .select("id")
        .eq("name", pet_type)
        .single()
        .execute()
    )

    return response.data['id']

def get_pet_status_id(pet_status:str):
    response = (
        supabase.table("pet_status")
        .select("id")
        .eq("name", pet_status)
        .single()
        .execute()
    )

    return response.data['id']

def get_owner_id(email:str):
    response = (
        supabase.table("profiles")
        .select("id")
        .eq("email", email)
        .single()
        .execute()
    )

    return response.data['id']

def get_municipality_id():
    response = (
        supabase.table("mao")
        .select("id")
        .eq("id", "84d41eb0-55bc-4efd-a518-1bb684aea68b")
        .single()
        .execute()
    )

    return response.data['id']

if __name__ == "__main__":
    print(get_municipality_id("mabitac"))