using UnityEngine;

public class PlayerController : MonoBehaviour
{
    public float speed = 10.0f;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    //https://discussions.unity.com/t/implementing-drawing-on-a-flat-surface-of-a-3d-object-looking-for-advice-943694/943694/4

    void Draw(){
        var ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;

        if(Physics.Raycast(ray, out hit, 10f))
        {
            // Get the texture off of whatever was hit
            Debug.Log("Hit: " + hit.point);
            Vector2 uv = hit.textureCoord;
            Renderer rend = hit.transform.GetComponent<Renderer>();
            Texture2D tex = rend.material.mainTexture as Texture2D;

            // Change the color of the texture at those UV points
            if (tex != null) {
                int pixelX = (int)(uv.x * tex.width);
                int pixelY = (int)(uv.y * tex.height);
                
                Color color = tex.GetPixel(pixelX, pixelY);
                Debug.Log("Pixel Color: " + color);
            }

            // Update the texture
        }
    }

    // Update is called once per frame
    void Update()
    {
        float translation = Input.GetAxis("Vertical") * speed * Time.deltaTime;
        float rotation = Input.GetAxis("Horizontal") * speed * Time.deltaTime;

        transform.Translate(0, 0, translation);
        transform.Rotate(0, rotation, 0);

        // listen for drawing
        if(Input.GetMouseButton(0)){
            Draw();
        }
    }
}
