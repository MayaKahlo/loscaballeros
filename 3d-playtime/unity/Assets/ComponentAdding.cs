using UnityEngine;


public class ComponentAdding : MonoBehaviour
{

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        Transform[] allChildren = this.gameObject.GetComponentsInChildren<Transform>(true);

        foreach (Transform child in allChildren)
        {
            if (child.transform.childCount > 0) continue;

            if (child.gameObject.GetComponent<MeshCollider>() == null &&
            child.gameObject.GetComponent<BoxCollider>() == null)
            {
                child.gameObject.AddComponent<MeshCollider>();
            }
        }

        Debug.Log("Added mesh colliders to all gameobjects");
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
