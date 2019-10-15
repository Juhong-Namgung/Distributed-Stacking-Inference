package dke.cs.knu;

public class PreProcessor {

    public float[][] convert(String input) {
        float[][] result = new float[1][21];
        String[] token = input.split(",", -1);
        for (int i = 0; i < 21; i++) {
            result[0][i] = Float.parseFloat(token[i]);
        }
    return result;
    }
}
