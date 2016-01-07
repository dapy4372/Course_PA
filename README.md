trainLSTMandMLP.py:
    parser.add_argument('-idim', '--image_feature_dim', type=int, default=4096)
    parser.add_argument('-ldim', '--language_feature_dim', type=int, default=300)
    parser.add_argument('-qf', '--question_feature', type=str, required=True)
    parser.add_argument('-cf', '--choice_feature', type=str, required=True)
    parser.add_argument('-lstm', type=bool, default=False)
    parser.add_argument('-lstm_units', type=int, default=512)
    parser.add_argument('-lstm_layers', type=int, default=1)
    parser.add_argument('-u', '--mlp_units', nargs='+', type=int, required=True)
    parser.add_argument('-a', '--mlp_activation', type=str, default='softplus')
    parser.add_argument('-odim', '--mlp_output_dim', type=int, default=300)
    parser.add_argument('-dropout', type=float, default=1.0)
    parser.add_argument('-maxout', type=bool, default=False)
    parser.add_argument('-memory_limit', type=float, default=6.0)
    parser.add_argument('-cross_valid', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-epochs', type=int, default=100)


===
evaluateLSTMandMLP.py:
    parser.add_argument('-predict_type', type=str, default='test')
    parser.add_argument('-language_feature_dim', type=int, default=300)
    parser.add_argument('-model', type=str, required=True)
    parser.add_argument('-weights', type=str, required=True)
    parser.add_argument('-question_feature', type=str, required=True)
    parser.add_argument('-choice_feature', type=str, required=True)
