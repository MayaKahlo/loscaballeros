using UnityEngine;

public class PlayerController : MonoBehaviour
{
    public float speed = 10.0f;
    Color drawColor = Color.green;
    bool inLine = false;
    LineRenderer currentLine;
    float timer;

    void Draw(){
        var ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;

        if(Physics.Raycast(ray, out hit, 30f))
        {
            if(!inLine){
                // Start new line
                GameObject newLineObj = new GameObject();
                currentLine = newLineObj.AddComponent<LineRenderer>();
                currentLine.material = new Material(Shader.Find("Sprites/Default"));
                currentLine.startColor = drawColor;
                currentLine.endColor = drawColor;
                currentLine.startWidth = 0.2f;
                currentLine.endWidth = 0.2f;
                currentLine.positionCount = 1;
                // send point in the direction of the normal a little bit
                currentLine.SetPosition(0, hit.point + 0.1f*hit.normal);

                inLine = true;
            }
            else{
                // Add to current linerenderer
                currentLine.positionCount++;
                currentLine.SetPosition(currentLine.positionCount - 1, hit.point + 0.1f*hit.normal);
            }
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
            timer = 0.5f;
        }
        else if(inLine && timer <= 0){
            // Stop line
            inLine = false;
        }

        timer -= Time.deltaTime;
    }
}
