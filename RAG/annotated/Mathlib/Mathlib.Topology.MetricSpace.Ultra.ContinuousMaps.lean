/-- Continuous maps from a compact space to an ultrametric space are an ultrametric space. -/
instance ContinuousMap.isUltrametricDist {X Y : Type*}
    [TopologicalSpace X] [CompactSpace X] [MetricSpace Y] [IsUltrametricDist Y] :
    IsUltrametricDist C(X, Y) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : CompactSpace X
    inst✝¹ : MetricSpace Y
    inst✝ : IsUltrametricDist Y
    ⊢ IsUltrametricDist (ContinuousMap X Y)
  -/
  constructor
  /-
    case dist_triangle_max
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : CompactSpace X
    inst✝¹ : MetricSpace Y
    inst✝ : IsUltrametricDist Y
    ⊢ ∀ (x y z : ContinuousMap X Y), LE.le (Dist.dist x z) (Max.max (Dist.dist x y …
  -/
  intro f g h
  /-
    case dist_triangle_max
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : CompactSpace X
    inst✝¹ : MetricSpace Y
    inst✝ : IsUltrametricDist Y
    f g h : ContinuousMap X Y
    ⊢ LE.le (Dist.dist f h) (Max.max (Dist.dist f g) (Dist.dist g h))
  -/
  rw [ContinuousMap.dist_le (by positivity)]
  /-
    case dist_triangle_max
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : CompactSpace X
    inst✝¹ : MetricSpace Y
    inst✝ : IsUltrametricDist Y
    f g h : ContinuousMap X Y
    ⊢ ∀ (x : X), LE.le (Dist.dist (f x) (h x)) (Max.max (Dist.dist f g) (Dist.dist …
  -/
  refine fun x ↦ (dist_triangle_max (f x) (g x) (h x)).trans (max_le_max ?_ ?_) <;>
  /-
    case dist_triangle_max.refine_1
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : CompactSpace X
    inst✝¹ : MetricSpace Y
    inst✝ : IsUltrametricDist Y
    f g h : ContinuousMap X Y
    x : X
    ⊢ LE.le (Dist.dist (f x) (g x)) (Dist.dist f g)
  -/
  /-
    🎉 no goals
  -/
  exact ContinuousMap.dist_apply_le_dist x
  /-
    🎉 no goals
  -/

