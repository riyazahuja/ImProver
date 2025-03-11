/-- The `dist : X → X → ℝ` respects the ultrametric inequality
of `dist(x, z) ≤ max (dist(x,y)) (dist(y,z))`. -/
class IsUltrametricDist (X : Type*) [Dist X] : Prop where
  dist_triangle_max : ∀ x y z : X, dist x z ≤ max (dist x y) (dist y z)


lemma dist_triangle_max : dist x z ≤ max (dist x y) (dist y z) :=
  IsUltrametricDist.dist_triangle_max x y z


/-- All triangles are isosceles in an ultrametric space. -/
lemma dist_eq_max_of_dist_ne_dist (h : dist x y ≠ dist y z) :
    dist x z = max (dist x y) (dist y z) := by
  /-
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x y z : X
    h : Ne (Dist.dist x y) (Dist.dist y z)
    ⊢ Eq (Dist.dist x z) (Max.max (Dist.dist x y) (Dist.dist y z))
  -/
  apply le_antisymm (dist_triangle_max x y z)
  /-
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x y z : X
    h : Ne (Dist.dist x y) (Dist.dist y z)
    ⊢ LE.le (Max.max (Dist.dist x y) (Dist.dist y z)) (Dist.dist x z)
  -/
  rcases h.lt_or_lt with h | h
    /-
      case inl
      X : Type u_1
      inst✝¹ : PseudoMetricSpace X
      inst✝ : IsUltrametricDist X
      x y z : X
      h✝ : Ne (Dist.dist x y) (Dist.dist y z)
      h : LT.lt (Dist.dist x y) (Dist.dist y z)
      ⊢ LE.le (Max.max (Dist.dist x y) (Dist.dist y z)) (Dist.dist x z)
    -/
  · rw [max_eq_right h.le]
    /-
      case inl
      X : Type u_1
      inst✝¹ : PseudoMetricSpace X
      inst✝ : IsUltrametricDist X
      x y z : X
      h✝ : Ne (Dist.dist x y) (Dist.dist y z)
      h : LT.lt (Dist.dist x y) (Dist.dist y z)
      ⊢ LE.le (Dist.dist y z) (Dist.dist x z)
    -/
    apply (le_max_iff.mp <| dist_triangle_max y x z).resolve_left
    /-
      case inl
      X : Type u_1
      inst✝¹ : PseudoMetricSpace X
      inst✝ : IsUltrametricDist X
      x y z : X
      h✝ : Ne (Dist.dist x y) (Dist.dist y z)
      h : LT.lt (Dist.dist x y) (Dist.dist y z)
      ⊢ Not (LE.le (Dist.dist y z) (Dist.dist y x))
    -/
    simpa only [not_le, dist_comm x y] using h
    /-
      🎉 no goals
    -/
    /-
      case inr
      X : Type u_1
      inst✝¹ : PseudoMetricSpace X
      inst✝ : IsUltrametricDist X
      x y z : X
      h✝ : Ne (Dist.dist x y) (Dist.dist y z)
      h : LT.lt (Dist.dist y z) (Dist.dist x y)
      ⊢ LE.le (Max.max (Dist.dist x y) (Dist.dist y z)) (Dist.dist x z)
    -/
  · rw [max_eq_left h.le, dist_comm x y, dist_comm x z]
    /-
      case inr
      X : Type u_1
      inst✝¹ : PseudoMetricSpace X
      inst✝ : IsUltrametricDist X
      x y z : X
      h✝ : Ne (Dist.dist x y) (Dist.dist y z)
      h : LT.lt (Dist.dist y z) (Dist.dist x y)
      ⊢ LE.le (Dist.dist y x) (Dist.dist z x)
    -/
    apply (le_max_iff.mp <| dist_triangle_max y z x).resolve_left
    /-
      case inr
      X : Type u_1
      inst✝¹ : PseudoMetricSpace X
      inst✝ : IsUltrametricDist X
      x y z : X
      h✝ : Ne (Dist.dist x y) (Dist.dist y z)
      h : LT.lt (Dist.dist y z) (Dist.dist x y)
      ⊢ Not (LE.le (Dist.dist y x) (Dist.dist y z))
    -/
    simpa only [not_le, dist_comm x y] using h
    /-
      🎉 no goals
    -/


instance subtype (p : X → Prop) : IsUltrametricDist (Subtype p) :=
                  /-
                    X : Type u_1
                    inst✝¹ : PseudoMetricSpace X
                    inst✝ : IsUltrametricDist X
                    x y z : X
                    r s : Real
                    p : X → Prop
                    x✝² x✝¹ x✝ : Subtype p
                    ⊢ LE.le (Dist.dist x✝² x✝) (Max.max (Dist.dist x✝² x✝¹) (Dist.dist x✝¹ x✝))
                  -/
  ⟨fun _ _ _ ↦ by simpa [Subtype.dist_eq] using dist_triangle_max _ _ _⟩
                  /-
                    🎉 no goals
                  -/


lemma ball_eq_of_mem {x y : X} {r : ℝ} (h : y ∈ ball x r) : ball x r = ball y r := by
  /-
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x y : X
    r : Real
    h : Membership.mem (Metric.ball x r) y
    ⊢ Eq (Metric.ball x r) (Metric.ball y r)
  -/
  ext a
  /-
    case h
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x y : X
    r : Real
    h : Membership.mem (Metric.ball x r) y
    a : X
    ⊢ Iff (Membership.mem (Metric.ball x r) a) (Membership.mem (Metric.ball y r) a)
  -/
  simp_rw [mem_ball] at h ⊢
  /-
    case h
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x y : X
    r : Real
    a : X
    h : LT.lt (Dist.dist y x) r
    ⊢ Iff (LT.lt (Dist.dist a x) r) (LT.lt (Dist.dist a y) r)
  -/
  constructor <;> intro h' <;>
  /-
    case h.mp
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x y : X
    r : Real
    a : X
    h : LT.lt (Dist.dist y x) r
    h' : LT.lt (Dist.dist a x) r
    ⊢ LT.lt (Dist.dist a y) r
  -/
  /-
    🎉 no goals
  -/
  exact (dist_triangle_max _ _ _).trans_lt (max_lt h' (dist_comm x _ ▸ h))
  /-
    🎉 no goals
  -/


lemma mem_ball_iff {x y : X} {r : ℝ} : y ∈ ball x r ↔ x ∈ ball y r := by
  cases lt_or_le 0 r with
  | inl hr =>
    constructor <;> intro h <;>
    rw [← ball_eq_of_mem h] <;>
    simp [hr]
  | inr hr => simp [ball_eq_empty.mpr hr]


lemma ball_subset_trichotomy :
    ball x r ⊆ ball y s ∨ ball y s ⊆ ball x r ∨ Disjoint (ball x r) (ball y s) := by
  /-
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x y : X
    r s : Real
    ⊢ Or (HasSubset.Subset (Metric.ball x r) (Metric.ball y s)) (Or (HasSubset.Sub …
  -/
  wlog hrs : r ≤ s generalizing x y r s
    /-
      case inr
      X : Type u_1
      inst✝¹ : PseudoMetricSpace X
      inst✝ : IsUltrametricDist X
      x y : X
      r s : Real
      this : ∀ (x y : X) (r s : Real), LE.le r s → Or (HasSubset.Subset (Metric.ball …
      hrs : Not (LE.le r s)
      ⊢ Or (HasSubset.Subset (Metric.ball x r) (Metric.ball y s)) (Or (HasSubset.Sub …
    -/
  · rw [disjoint_comm, ← or_assoc, or_comm (b := _ ⊆ _), or_assoc]
    /-
      case inr
      X : Type u_1
      inst✝¹ : PseudoMetricSpace X
      inst✝ : IsUltrametricDist X
      x y : X
      r s : Real
      this : ∀ (x y : X) (r s : Real), LE.le r s → Or (HasSubset.Subset (Metric.ball …
      hrs : Not (LE.le r s)
      ⊢ Or (HasSubset.Subset (Metric.ball y s) (Metric.ball x r)) (Or (HasSubset.Sub …
    -/
    exact this y x s r (lt_of_not_le hrs).le
    /-
      🎉 no goals
    -/
    /-
      X : Type u_1
      inst✝¹ : PseudoMetricSpace X
      inst✝ : IsUltrametricDist X
      x✝ y✝ : X
      r✝ s✝ : Real
      x y : X
      r s : Real
      hrs : LE.le r s
      ⊢ Or (HasSubset.Subset (Metric.ball x r) (Metric.ball y s)) (Or (HasSubset.Sub …
    -/
  · refine Set.disjoint_or_nonempty_inter (ball x r) (ball y s) |>.symm.imp (fun h ↦ ?_) (Or.inr ·)
    /-
      X : Type u_1
      inst✝¹ : PseudoMetricSpace X
      inst✝ : IsUltrametricDist X
      x✝ y✝ : X
      r✝ s✝ : Real
      x y : X
      r s : Real
      hrs : LE.le r s
      h : (Inter.inter (Metric.ball x r) (Metric.ball y s)).Nonempty
      ⊢ HasSubset.Subset (Metric.ball x r) (Metric.ball y s)
    -/
    obtain ⟨hxz, hyz⟩ := (Set.mem_inter_iff _ _ _).mp h.some_mem
    /-
      case intro
      X : Type u_1
      inst✝¹ : PseudoMetricSpace X
      inst✝ : IsUltrametricDist X
      x✝ y✝ : X
      r✝ s✝ : Real
      x y : X
      r s : Real
      hrs : LE.le r s
      h : (Inter.inter (Metric.ball x r) (Metric.ball y s)).Nonempty
      hxz : Membership.mem (Metric.ball x r) h.some
      hyz : Membership.mem (Metric.ball y s) h.some
      ⊢ HasSubset.Subset (Metric.ball x r) (Metric.ball y s)
    -/
    have hx := ball_subset_ball hrs (x := x)
    /-
      case intro
      X : Type u_1
      inst✝¹ : PseudoMetricSpace X
      inst✝ : IsUltrametricDist X
      x✝ y✝ : X
      r✝ s✝ : Real
      x y : X
      r s : Real
      hrs : LE.le r s
      h : (Inter.inter (Metric.ball x r) (Metric.ball y s)).Nonempty
      hxz : Membership.mem (Metric.ball x r) h.some
      hyz : Membership.mem (Metric.ball y s) h.some
      hx : HasSubset.Subset (Metric.ball x r) (Metric.ball x s)
      ⊢ HasSubset.Subset (Metric.ball x r) (Metric.ball y s)
    -/
    rwa [ball_eq_of_mem hyz |>.trans (ball_eq_of_mem <| hx hxz).symm]
    /-
      🎉 no goals
    -/


lemma ball_eq_or_disjoint :
    ball x r = ball y r ∨ Disjoint (ball x r) (ball y r) := by
  /-
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x y : X
    r : Real
    ⊢ Or (Eq (Metric.ball x r) (Metric.ball y r)) (Disjoint (Metric.ball x r) (Met …
  -/
  refine Set.disjoint_or_nonempty_inter (ball x r) (ball y r) |>.symm.imp (fun h ↦ ?_) id
  /-
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x y : X
    r : Real
    h : (Inter.inter (Metric.ball x r) (Metric.ball y r)).Nonempty
    ⊢ Eq (Metric.ball x r) (Metric.ball y r)
  -/
  have h₁ := ball_eq_of_mem <| Set.inter_subset_left h.some_mem
  /-
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x y : X
    r : Real
    h : (Inter.inter (Metric.ball x r) (Metric.ball y r)).Nonempty
    h₁ : Eq (Metric.ball x r) (Metric.ball h.some r)
    ⊢ Eq (Metric.ball x r) (Metric.ball y r)
  -/
  have h₂ := ball_eq_of_mem <| Set.inter_subset_right h.some_mem
  /-
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x y : X
    r : Real
    h : (Inter.inter (Metric.ball x r) (Metric.ball y r)).Nonempty
    h₁ : Eq (Metric.ball x r) (Metric.ball h.some r)
    h₂ : Eq (Metric.ball y r) (Metric.ball h.some r)
    ⊢ Eq (Metric.ball x r) (Metric.ball y r)
  -/
  exact h₁.trans h₂.symm
  /-
    🎉 no goals
  -/


lemma closedBall_eq_of_mem {x y: X} {r : ℝ} (h : y ∈ closedBall x r) :
    closedBall x r = closedBall y r := by
  /-
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x y : X
    r : Real
    h : Membership.mem (Metric.closedBall x r) y
    ⊢ Eq (Metric.closedBall x r) (Metric.closedBall y r)
  -/
  ext
  /-
    case h
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x y : X
    r : Real
    h : Membership.mem (Metric.closedBall x r) y
    x✝ : X
    ⊢ Iff (Membership.mem (Metric.closedBall x r) x✝) (Membership.mem (Metric.clos …
  -/
  simp_rw [mem_closedBall] at h ⊢
  /-
    case h
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x y : X
    r : Real
    x✝ : X
    h : LE.le (Dist.dist y x) r
    ⊢ Iff (LE.le (Dist.dist x✝ x) r) (LE.le (Dist.dist x✝ y) r)
  -/
  constructor <;> intro h' <;>
  /-
    case h.mp
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x y : X
    r : Real
    x✝ : X
    h : LE.le (Dist.dist y x) r
    h' : LE.le (Dist.dist x✝ x) r
    ⊢ LE.le (Dist.dist x✝ y) r
  -/
  /-
    🎉 no goals
  -/
  exact (dist_triangle_max _ _ _).trans (max_le h' (dist_comm x _ ▸ h))
  /-
    🎉 no goals
  -/


lemma mem_closedBall_iff {x y: X} {r : ℝ} :
    y ∈ closedBall x r ↔ x ∈ closedBall y r := by
  cases le_or_lt 0 r with
  | inl hr =>
    constructor <;> intro h <;>
    rw [← closedBall_eq_of_mem h] <;>
    simp [hr]
  | inr hr => simp [closedBall_eq_empty.mpr hr]


lemma closedBall_subset_trichotomy :
    closedBall x r ⊆ closedBall y s ∨ closedBall y s ⊆ closedBall x r ∨
    Disjoint (closedBall x r) (closedBall y s) := by
  /-
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x y : X
    r s : Real
    ⊢ Or (HasSubset.Subset (Metric.closedBall x r) (Metric.closedBall y s)) (Or (H …
  -/
  wlog hrs : r ≤ s generalizing x y r s
    /-
      case inr
      X : Type u_1
      inst✝¹ : PseudoMetricSpace X
      inst✝ : IsUltrametricDist X
      x y : X
      r s : Real
      this : ∀ (x y : X) (r s : Real), LE.le r s → Or (HasSubset.Subset (Metric.clos …
      hrs : Not (LE.le r s)
      ⊢ Or (HasSubset.Subset (Metric.closedBall x r) (Metric.closedBall y s)) (Or (H …
    -/
  · rw [disjoint_comm, ← or_assoc, or_comm (b := _ ⊆ _), or_assoc]
    /-
      case inr
      X : Type u_1
      inst✝¹ : PseudoMetricSpace X
      inst✝ : IsUltrametricDist X
      x y : X
      r s : Real
      this : ∀ (x y : X) (r s : Real), LE.le r s → Or (HasSubset.Subset (Metric.clos …
      hrs : Not (LE.le r s)
      ⊢ Or (HasSubset.Subset (Metric.closedBall y s) (Metric.closedBall x r)) (Or (H …
    -/
    exact this y x s r (lt_of_not_le hrs).le
    /-
      🎉 no goals
    -/
  · refine Set.disjoint_or_nonempty_inter (closedBall x r) (closedBall y s) |>.symm.imp
      (fun h ↦ ?_) (Or.inr ·)
    /-
      X : Type u_1
      inst✝¹ : PseudoMetricSpace X
      inst✝ : IsUltrametricDist X
      x✝ y✝ : X
      r✝ s✝ : Real
      x y : X
      r s : Real
      hrs : LE.le r s
      h : (Inter.inter (Metric.closedBall x r) (Metric.closedBall y s)).Nonempty
      ⊢ HasSubset.Subset (Metric.closedBall x r) (Metric.closedBall y s)
    -/
    obtain ⟨hxz, hyz⟩ := (Set.mem_inter_iff _ _ _).mp h.some_mem
    /-
      case intro
      X : Type u_1
      inst✝¹ : PseudoMetricSpace X
      inst✝ : IsUltrametricDist X
      x✝ y✝ : X
      r✝ s✝ : Real
      x y : X
      r s : Real
      hrs : LE.le r s
      h : (Inter.inter (Metric.closedBall x r) (Metric.closedBall y s)).Nonempty
      hxz : Membership.mem (Metric.closedBall x r) h.some
      hyz : Membership.mem (Metric.closedBall y s) h.some
      ⊢ HasSubset.Subset (Metric.closedBall x r) (Metric.closedBall y s)
    -/
    have hx := closedBall_subset_closedBall hrs (x := x)
    /-
      case intro
      X : Type u_1
      inst✝¹ : PseudoMetricSpace X
      inst✝ : IsUltrametricDist X
      x✝ y✝ : X
      r✝ s✝ : Real
      x y : X
      r s : Real
      hrs : LE.le r s
      h : (Inter.inter (Metric.closedBall x r) (Metric.closedBall y s)).Nonempty
      hxz : Membership.mem (Metric.closedBall x r) h.some
      hyz : Membership.mem (Metric.closedBall y s) h.some
      hx : HasSubset.Subset (Metric.closedBall x r) (Metric.closedBall x s)
      ⊢ HasSubset.Subset (Metric.closedBall x r) (Metric.closedBall y s)
    -/
    rwa [closedBall_eq_of_mem hyz |>.trans (closedBall_eq_of_mem <| hx hxz).symm]
    /-
      🎉 no goals
    -/


lemma isClosed_ball (x : X) (r : ℝ) : IsClosed (ball x r) := by
  cases le_or_lt r 0 with
  | inl hr =>
    simp [ball_eq_empty.mpr hr]
  | inr h =>
    rw [← isOpen_compl_iff, isOpen_iff]
    simp only [Set.mem_compl_iff, not_lt, gt_iff_lt]
    intro y hy
    cases ball_eq_or_disjoint x y r with
    | inl hd =>
      rw [hd] at hy
      simp [h.not_le] at hy
    | inr hd =>
      use r
      simp [h, hy, ← Set.le_iff_subset, le_compl_iff_disjoint_left, hd]


lemma isClopen_ball : IsClopen (ball x r) := ⟨isClosed_ball x r, isOpen_ball⟩


lemma frontier_ball_eq_empty : frontier (ball x r) = ∅ :=
  isClopen_iff_frontier_eq_empty.mp (isClopen_ball x r)


lemma closedBall_eq_or_disjoint :
    closedBall x r = closedBall y r ∨ Disjoint (closedBall x r) (closedBall y r) := by
  refine Set.disjoint_or_nonempty_inter (closedBall x r) (closedBall y r) |>.symm.imp
    (fun h ↦ ?_) id
  /-
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x y : X
    r : Real
    h : (Inter.inter (Metric.closedBall x r) (Metric.closedBall y r)).Nonempty
    ⊢ Eq (Metric.closedBall x r) (Metric.closedBall y r)
  -/
  have h₁ := closedBall_eq_of_mem <| Set.inter_subset_left h.some_mem
  /-
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x y : X
    r : Real
    h : (Inter.inter (Metric.closedBall x r) (Metric.closedBall y r)).Nonempty
    h₁ : Eq (Metric.closedBall x r) (Metric.closedBall h.some r)
    ⊢ Eq (Metric.closedBall x r) (Metric.closedBall y r)
  -/
  have h₂ := closedBall_eq_of_mem <| Set.inter_subset_right h.some_mem
  /-
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x y : X
    r : Real
    h : (Inter.inter (Metric.closedBall x r) (Metric.closedBall y r)).Nonempty
    h₁ : Eq (Metric.closedBall x r) (Metric.closedBall h.some r)
    h₂ : Eq (Metric.closedBall y r) (Metric.closedBall h.some r)
    ⊢ Eq (Metric.closedBall x r) (Metric.closedBall y r)
  -/
  exact h₁.trans h₂.symm
  /-
    🎉 no goals
  -/


lemma isOpen_closedBall {r : ℝ} (hr : r ≠ 0) : IsOpen (closedBall x r) := by
  cases lt_or_gt_of_ne hr with
  | inl h =>
    simp [closedBall_eq_empty.mpr h]
  | inr h =>
    rw [isOpen_iff]
    simp only [Set.mem_compl_iff, not_lt, gt_iff_lt]
    intro y hy
    cases closedBall_eq_or_disjoint x y r with
    | inl hd =>
      use r
      simp [h, hd, ball_subset_closedBall]
    | inr hd =>
      simp [closedBall_eq_of_mem hy, h.not_lt] at hd


lemma isClopen_closedBall {r : ℝ} (hr : r ≠ 0) : IsClopen (closedBall x r) :=
  ⟨Metric.isClosed_ball, isOpen_closedBall x hr⟩


lemma frontier_closedBall_eq_empty {r : ℝ} (hr : r ≠ 0) : frontier (closedBall x r) = ∅ :=
  isClopen_iff_frontier_eq_empty.mp (isClopen_closedBall x hr)


lemma isOpen_sphere {r : ℝ} (hr : r ≠ 0) : IsOpen (sphere x r) := by
  /-
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x : X
    r : Real
    hr : Ne r 0
    ⊢ IsOpen (Metric.sphere x r)
  -/
  rw [← closedBall_diff_ball, sdiff_eq]
  /-
    X : Type u_1
    inst✝¹ : PseudoMetricSpace X
    inst✝ : IsUltrametricDist X
    x : X
    r : Real
    hr : Ne r 0
    ⊢ IsOpen (Min.min (Metric.closedBall x r) (HasCompl.compl (Metric.ball x r)))
  -/
  exact (isOpen_closedBall x hr).inter (isClosed_ball x r).isOpen_compl
  /-
    🎉 no goals
  -/


lemma isClopen_sphere {r : ℝ} (hr : r ≠ 0) : IsClopen (sphere x r) :=
  ⟨Metric.isClosed_sphere, isOpen_sphere x hr⟩


