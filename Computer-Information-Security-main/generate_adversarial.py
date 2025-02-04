import torch
from torch.autograd import grad
from point_rcnn_lib import PointRCNN  # Assuming correct import based on setup

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    if points.shape[0] > 1024:
        # Sample or pad the points if necessary
        points = points[:1024]
    return torch.tensor(points, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

def iter_grad_op(x, model, labels, iterations=400, epsilon=0.01, norm='inf', clip_min=None, clip_max=None):
    alpha = epsilon / float(iterations)
    x_adv = x.clone().detach().requires_grad_(True)
    
    for _ in range(iterations):
        outputs = model(x_adv)
        loss = outputs.sum()  # Simplified loss calculation
        loss.backward()
        with torch.no_grad():
            if norm == "inf":
                perturbations = alpha * x_adv.grad.sign()
            elif norm == "2":
                perturbations = alpha * x_adv.grad / (x_adv.grad.norm(2, dim=-1, keepdim=True) + 1e-6)
            perturbations[:, :, :3] = 0  # Zero out perturbations for x, y, z
            x_adv += perturbations
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
            x_adv.grad.zero_()
    
    return x_adv.detach()

def main():
    model = PointRCNN.load_from_checkpoint('path_to_checkpoint.pth')
    model.eval()

    file_path = 'path_to_your_target_object_pcd.pcd'  # Path to the PCD file of the target object
    x_data = load_point_cloud(file_path)
    labels = torch.randint(0, 10, (1,))  # Assuming a dummy label for demonstration

    adversarial_data = iter_grad_op(x_data, model, labels)
    torch.save(adversarial_data, 'adversarial_data.pt')


if __name__ == '__main__':
    main()
