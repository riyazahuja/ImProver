/-- Define a predistance on `X ⊕ Y`, for which `Φ p` and `Ψ p` are at distance `ε` -/
def glueDist (Φ : Z → X) (Ψ : Z → Y) (ε : ℝ) : X ⊕ Y → X ⊕ Y → ℝ
  | .inl x, .inl y => dist x y
  | .inr x, .inr y => dist x y
  | .inl x, .inr y => (⨅ p, dist x (Φ p) + dist y (Ψ p)) + ε
  | .inr x, .inl y => (⨅ p, dist y (Φ p) + dist x (Ψ p)) + ε


private theorem glueDist_self (Φ : Z → X) (Ψ : Z → Y) (ε : ℝ) : ∀ x, glueDist Φ Ψ ε x x = 0
  | .inl _ => dist_self _
  | .inr _ => dist_self _


theorem glueDist_glued_points [Nonempty Z] (Φ : Z → X) (Ψ : Z → Y) (ε : ℝ) (p : Z) :
    glueDist Φ Ψ ε (.inl (Φ p)) (.inr (Ψ p)) = ε := by
  have : ⨅ q, dist (Φ p) (Φ q) + dist (Ψ p) (Ψ q) = 0 := by
    have A : ∀ q, 0 ≤ dist (Φ p) (Φ q) + dist (Ψ p) (Ψ q) := fun _ =>
      add_nonneg dist_nonneg dist_nonneg
    refine le_antisymm ?_ (le_ciInf A)
    have : 0 = dist (Φ p) (Φ p) + dist (Ψ p) (Ψ p) := by simp
    rw [this]
    exact ciInf_le ⟨0, forall_mem_range.2 A⟩ p
  /-
    X : Type u
    Y : Type v
    Z : Type w
    inst✝² : MetricSpace X
    inst✝¹ : MetricSpace Y
    inst✝ : Nonempty Z
    Φ : Z → X
    Ψ : Z → Y
    ε : Real
    p : Z
    this : Eq (iInf fun q => HAdd.hAdd (Dist.dist (Φ p) (Φ q)) (Dist.dist (Ψ p) (Ψ …
    ⊢ Eq (Metric.glueDist Φ Ψ ε (Sum.inl (Φ p)) (Sum.inr (Ψ p))) ε
  -/
  simp only [glueDist, this, zero_add]
  /-
    🎉 no goals
  -/


private theorem glueDist_comm (Φ : Z → X) (Ψ : Z → Y) (ε : ℝ) :
    ∀ x y, glueDist Φ Ψ ε x y = glueDist Φ Ψ ε y x
  | .inl _, .inl _ => dist_comm _ _
  | .inr _, .inr _ => dist_comm _ _
  | .inl _, .inr _ => rfl
  | .inr _, .inl _ => rfl


theorem glueDist_swap (Φ : Z → X) (Ψ : Z → Y) (ε : ℝ) :
    ∀ x y, glueDist Ψ Φ ε x.swap y.swap = glueDist Φ Ψ ε x y
  | .inl _, .inl _ => rfl
  | .inr _, .inr _ => rfl
                         /-
                           X : Type u
                           Y : Type v
                           Z : Type w
                           inst✝¹ : MetricSpace X
                           inst✝ : MetricSpace Y
                           Φ : Z → X
                           Ψ : Z → Y
                           ε : Real
                           val✝¹ : X
                           val✝ : Y
                           ⊢ Eq (Metric.glueDist Ψ Φ ε (Sum.inl val✝¹).swap (Sum.inr val✝).swap) (Metric. …
                         -/
  | .inl _, .inr _ => by simp only [glueDist, Sum.swap_inl, Sum.swap_inr, dist_comm, add_comm]
                         /-
                           🎉 no goals
                         -/
                         /-
                           X : Type u
                           Y : Type v
                           Z : Type w
                           inst✝¹ : MetricSpace X
                           inst✝ : MetricSpace Y
                           Φ : Z → X
                           Ψ : Z → Y
                           ε : Real
                           val✝¹ : Y
                           val✝ : X
                           ⊢ Eq (Metric.glueDist Ψ Φ ε (Sum.inr val✝¹).swap (Sum.inl val✝).swap) (Metric. …
                         -/
  | .inr _, .inl _ => by simp only [glueDist, Sum.swap_inl, Sum.swap_inr, dist_comm, add_comm]
                         /-
                           🎉 no goals
                         -/


theorem le_glueDist_inl_inr (Φ : Z → X) (Ψ : Z → Y) (ε : ℝ) (x y) :
    ε ≤ glueDist Φ Ψ ε (.inl x) (.inr y) :=
  le_add_of_nonneg_left <| Real.iInf_nonneg fun _ => add_nonneg dist_nonneg dist_nonneg


theorem le_glueDist_inr_inl (Φ : Z → X) (Ψ : Z → Y) (ε : ℝ) (x y) :
    ε ≤ glueDist Φ Ψ ε (.inr x) (.inl y) := by
  /-
    X : Type u
    Y : Type v
    Z : Type w
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    Φ : Z → X
    Ψ : Z → Y
    ε : Real
    x : Y
    y : X
    ⊢ LE.le ε (Metric.glueDist Φ Ψ ε (Sum.inr x) (Sum.inl y))
  -/
  rw [glueDist_comm]; apply le_glueDist_inl_inr
                      /-
                        🎉 no goals
                      -/


private theorem glueDist_triangle_inl_inr_inr (Φ : Z → X) (Ψ : Z → Y) (ε : ℝ) (x : X) (y z : Y) :
    glueDist Φ Ψ ε (.inl x) (.inr z) ≤
      glueDist Φ Ψ ε (.inl x) (.inr y) + glueDist Φ Ψ ε (.inr y) (.inr z) := by
  /-
    X : Type u
    Y : Type v
    Z : Type w
    inst✝² : MetricSpace X
    inst✝¹ : MetricSpace Y
    inst✝ : Nonempty Z
    Φ : Z → X
    Ψ : Z → Y
    ε : Real
    x : X
    y z : Y
    ⊢ LE.le (Metric.glueDist Φ Ψ ε (Sum.inl x) (Sum.inr z)) (HAdd.hAdd (Metric.glu …
  -/
  simp only [glueDist]
  /-
    X : Type u
    Y : Type v
    Z : Type w
    inst✝² : MetricSpace X
    inst✝¹ : MetricSpace Y
    inst✝ : Nonempty Z
    Φ : Z → X
    Ψ : Z → Y
    ε : Real
    x : X
    y z : Y
    ⊢ LE.le (HAdd.hAdd (iInf fun p => HAdd.hAdd (Dist.dist x (Φ p)) (Dist.dist z ( …
  -/
  rw [add_right_comm, add_le_add_iff_right]
  /-
    X : Type u
    Y : Type v
    Z : Type w
    inst✝² : MetricSpace X
    inst✝¹ : MetricSpace Y
    inst✝ : Nonempty Z
    Φ : Z → X
    Ψ : Z → Y
    ε : Real
    x : X
    y z : Y
    ⊢ LE.le (iInf fun p => HAdd.hAdd (Dist.dist x (Φ p)) (Dist.dist z (Ψ p))) (HAd …
  -/
  refine le_ciInf_add fun p => ciInf_le_of_le ⟨0, ?_⟩ p ?_
    /-
      case refine_1
      X : Type u
      Y : Type v
      Z : Type w
      inst✝² : MetricSpace X
      inst✝¹ : MetricSpace Y
      inst✝ : Nonempty Z
      Φ : Z → X
      Ψ : Z → Y
      ε : Real
      x : X
      y z : Y
      p : Z
      ⊢ Membership.mem (lowerBounds (Set.range fun p => HAdd.hAdd (Dist.dist x (Φ p) …
    -/
  · exact forall_mem_range.2 fun _ => add_nonneg dist_nonneg dist_nonneg
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u
      Y : Type v
      Z : Type w
      inst✝² : MetricSpace X
      inst✝¹ : MetricSpace Y
      inst✝ : Nonempty Z
      Φ : Z → X
      Ψ : Z → Y
      ε : Real
      x : X
      y z : Y
      p : Z
      ⊢ LE.le (HAdd.hAdd (Dist.dist x (Φ p)) (Dist.dist z (Ψ p))) (HAdd.hAdd (HAdd.h …
    -/
  · linarith [dist_triangle_left z (Ψ p) y]
    /-
      🎉 no goals
    -/


private theorem glueDist_triangle_inl_inr_inl (Φ : Z → X) (Ψ : Z → Y) (ε : ℝ)
    (H : ∀ p q, |dist (Φ p) (Φ q) - dist (Ψ p) (Ψ q)| ≤ 2 * ε) (x : X) (y : Y) (z : X) :
    glueDist Φ Ψ ε (.inl x) (.inl z) ≤
      glueDist Φ Ψ ε (.inl x) (.inr y) + glueDist Φ Ψ ε (.inr y) (.inl z) := by
  /-
    X : Type u
    Y : Type v
    Z : Type w
    inst✝² : MetricSpace X
    inst✝¹ : MetricSpace Y
    inst✝ : Nonempty Z
    Φ : Z → X
    Ψ : Z → Y
    ε : Real
    H : ∀ (p q : Z), LE.le (abs (HSub.hSub (Dist.dist (Φ p) (Φ q)) (Dist.dist (Ψ p …
    x : X
    y : Y
    z : X
    ⊢ LE.le (Metric.glueDist Φ Ψ ε (Sum.inl x) (Sum.inl z)) (HAdd.hAdd (Metric.glu …
  -/
  simp_rw [glueDist, add_add_add_comm _ ε, add_assoc]
  /-
    X : Type u
    Y : Type v
    Z : Type w
    inst✝² : MetricSpace X
    inst✝¹ : MetricSpace Y
    inst✝ : Nonempty Z
    Φ : Z → X
    Ψ : Z → Y
    ε : Real
    H : ∀ (p q : Z), LE.le (abs (HSub.hSub (Dist.dist (Φ p) (Φ q)) (Dist.dist (Ψ p …
    x : X
    y : Y
    z : X
    ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (iInf fun p => HAdd.hAdd (Dist.dist x (Φ p) …
  -/
  refine le_ciInf_add fun p => ?_
  /-
    X : Type u
    Y : Type v
    Z : Type w
    inst✝² : MetricSpace X
    inst✝¹ : MetricSpace Y
    inst✝ : Nonempty Z
    Φ : Z → X
    Ψ : Z → Y
    ε : Real
    H : ∀ (p q : Z), LE.le (abs (HSub.hSub (Dist.dist (Φ p) (Φ q)) (Dist.dist (Ψ p …
    x : X
    y : Y
    z : X
    p : Z
    ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (HAdd.hAdd (Dist.dist x (Φ p)) (Dist.dist y …
  -/
  rw [add_left_comm, add_assoc, ← two_mul]
  /-
    X : Type u
    Y : Type v
    Z : Type w
    inst✝² : MetricSpace X
    inst✝¹ : MetricSpace Y
    inst✝ : Nonempty Z
    Φ : Z → X
    Ψ : Z → Y
    ε : Real
    H : ∀ (p q : Z), LE.le (abs (HSub.hSub (Dist.dist (Φ p) (Φ q)) (Dist.dist (Ψ p …
    x : X
    y : Y
    z : X
    p : Z
    ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (iInf fun p => HAdd.hAdd (Dist.dist z (Φ p) …
  -/
  refine le_ciInf_add fun q => ?_
  /-
    X : Type u
    Y : Type v
    Z : Type w
    inst✝² : MetricSpace X
    inst✝¹ : MetricSpace Y
    inst✝ : Nonempty Z
    Φ : Z → X
    Ψ : Z → Y
    ε : Real
    H : ∀ (p q : Z), LE.le (abs (HSub.hSub (Dist.dist (Φ p) (Φ q)) (Dist.dist (Ψ p …
    x : X
    y : Y
    z : X
    p q : Z
    ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (HAdd.hAdd (Dist.dist z (Φ q)) (Dist.dist y …
  -/
  rw [dist_comm z]
  /-
    X : Type u
    Y : Type v
    Z : Type w
    inst✝² : MetricSpace X
    inst✝¹ : MetricSpace Y
    inst✝ : Nonempty Z
    Φ : Z → X
    Ψ : Z → Y
    ε : Real
    H : ∀ (p q : Z), LE.le (abs (HSub.hSub (Dist.dist (Φ p) (Φ q)) (Dist.dist (Ψ p …
    x : X
    y : Y
    z : X
    p q : Z
    ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (HAdd.hAdd (Dist.dist (Φ q) z) (Dist.dist y …
  -/
  linarith [dist_triangle4 x (Φ p) (Φ q) z, dist_triangle_left (Ψ p) (Ψ q) y, (abs_le.1 (H p q)).2]
  /-
    🎉 no goals
  -/


private theorem glueDist_triangle (Φ : Z → X) (Ψ : Z → Y) (ε : ℝ)
    (H : ∀ p q, |dist (Φ p) (Φ q) - dist (Ψ p) (Ψ q)| ≤ 2 * ε) :
    ∀ x y z, glueDist Φ Ψ ε x z ≤ glueDist Φ Ψ ε x y + glueDist Φ Ψ ε y z
  | .inl _, .inl _, .inl _ => dist_triangle _ _ _
  | .inr _, .inr _, .inr _ => dist_triangle _ _ _
  | .inr x, .inl y, .inl z => by
    /-
      X : Type u
      Y : Type v
      Z : Type w
      inst✝² : MetricSpace X
      inst✝¹ : MetricSpace Y
      inst✝ : Nonempty Z
      Φ : Z → X
      Ψ : Z → Y
      ε : Real
      H : ∀ (p q : Z), LE.le (abs (HSub.hSub (Dist.dist (Φ p) (Φ q)) (Dist.dist (Ψ p …
      x : Y
      y z : X
      ⊢ LE.le (Metric.glueDist Φ Ψ ε (Sum.inr x) (Sum.inl z)) (HAdd.hAdd (Metric.glu …
    -/
    simp only [← glueDist_swap Φ]
    /-
      X : Type u
      Y : Type v
      Z : Type w
      inst✝² : MetricSpace X
      inst✝¹ : MetricSpace Y
      inst✝ : Nonempty Z
      Φ : Z → X
      Ψ : Z → Y
      ε : Real
      H : ∀ (p q : Z), LE.le (abs (HSub.hSub (Dist.dist (Φ p) (Φ q)) (Dist.dist (Ψ p …
      x : Y
      y z : X
      ⊢ LE.le (Metric.glueDist Ψ Φ ε (Sum.inr x).swap (Sum.inl z).swap) (HAdd.hAdd ( …
    -/
    apply glueDist_triangle_inl_inr_inr
    /-
      🎉 no goals
    -/
  | .inr x, .inr y, .inl z => by
    /-
      X : Type u
      Y : Type v
      Z : Type w
      inst✝² : MetricSpace X
      inst✝¹ : MetricSpace Y
      inst✝ : Nonempty Z
      Φ : Z → X
      Ψ : Z → Y
      ε : Real
      H : ∀ (p q : Z), LE.le (abs (HSub.hSub (Dist.dist (Φ p) (Φ q)) (Dist.dist (Ψ p …
      x y : Y
      z : X
      ⊢ LE.le (Metric.glueDist Φ Ψ ε (Sum.inr x) (Sum.inl z)) (HAdd.hAdd (Metric.glu …
    -/
    simpa only [glueDist_comm, add_comm] using glueDist_triangle_inl_inr_inr _ _ _ z y x
    /-
      🎉 no goals
    -/
  | .inl x, .inl y, .inr z => by
    simpa only [← glueDist_swap Φ, glueDist_comm, add_comm, Sum.swap_inl, Sum.swap_inr]
      using glueDist_triangle_inl_inr_inr Ψ Φ ε z y x
  | .inl _, .inr _, .inr _ => glueDist_triangle_inl_inr_inr ..
  | .inl x, .inr y, .inl z => glueDist_triangle_inl_inr_inl Φ Ψ ε H x y z
  | .inr x, .inl y, .inr z => by
    /-
      X : Type u
      Y : Type v
      Z : Type w
      inst✝² : MetricSpace X
      inst✝¹ : MetricSpace Y
      inst✝ : Nonempty Z
      Φ : Z → X
      Ψ : Z → Y
      ε : Real
      H : ∀ (p q : Z), LE.le (abs (HSub.hSub (Dist.dist (Φ p) (Φ q)) (Dist.dist (Ψ p …
      x : Y
      y : X
      z : Y
      ⊢ LE.le (Metric.glueDist Φ Ψ ε (Sum.inr x) (Sum.inr z)) (HAdd.hAdd (Metric.glu …
    -/
    simp only [← glueDist_swap Φ]
    /-
      X : Type u
      Y : Type v
      Z : Type w
      inst✝² : MetricSpace X
      inst✝¹ : MetricSpace Y
      inst✝ : Nonempty Z
      Φ : Z → X
      Ψ : Z → Y
      ε : Real
      H : ∀ (p q : Z), LE.le (abs (HSub.hSub (Dist.dist (Φ p) (Φ q)) (Dist.dist (Ψ p …
      x : Y
      y : X
      z : Y
      ⊢ LE.le (Metric.glueDist Ψ Φ ε (Sum.inr x).swap (Sum.inr z).swap) (HAdd.hAdd ( …
    -/
    apply glueDist_triangle_inl_inr_inl
    /-
      case H
      X : Type u
      Y : Type v
      Z : Type w
      inst✝² : MetricSpace X
      inst✝¹ : MetricSpace Y
      inst✝ : Nonempty Z
      Φ : Z → X
      Ψ : Z → Y
      ε : Real
      H : ∀ (p q : Z), LE.le (abs (HSub.hSub (Dist.dist (Φ p) (Φ q)) (Dist.dist (Ψ p …
      x : Y
      y : X
      z : Y
      ⊢ ∀ (p q : Z), LE.le (abs (HSub.hSub (Dist.dist (Ψ p) (Ψ q)) (Dist.dist (Φ p)  …
    -/
    simpa only [abs_sub_comm]
    /-
      🎉 no goals
    -/


private theorem eq_of_glueDist_eq_zero (Φ : Z → X) (Ψ : Z → Y) (ε : ℝ) (ε0 : 0 < ε) :
    ∀ p q : X ⊕ Y, glueDist Φ Ψ ε p q = 0 → p = q
                            /-
                              X : Type u
                              Y : Type v
                              Z : Type w
                              inst✝¹ : MetricSpace X
                              inst✝ : MetricSpace Y
                              Φ : Z → X
                              Ψ : Z → Y
                              ε : Real
                              ε0 : LT.lt 0 ε
                              x y : X
                              h : Eq (Metric.glueDist Φ Ψ ε (Sum.inl x) (Sum.inl y)) 0
                              ⊢ Eq (Sum.inl x) (Sum.inl y)
                            -/
  | .inl x, .inl y, h => by rw [eq_of_dist_eq_zero h]
                            /-
                              🎉 no goals
                            -/
                            /-
                              X : Type u
                              Y : Type v
                              Z : Type w
                              inst✝¹ : MetricSpace X
                              inst✝ : MetricSpace Y
                              Φ : Z → X
                              Ψ : Z → Y
                              ε : Real
                              ε0 : LT.lt 0 ε
                              x : X
                              y : Y
                              h : Eq (Metric.glueDist Φ Ψ ε (Sum.inl x) (Sum.inr y)) 0
                              ⊢ Eq (Sum.inl x) (Sum.inr y)
                            -/
  | .inl x, .inr y, h => by exfalso; linarith [le_glueDist_inl_inr Φ Ψ ε x y]
                                     /-
                                       🎉 no goals
                                     -/
                            /-
                              X : Type u
                              Y : Type v
                              Z : Type w
                              inst✝¹ : MetricSpace X
                              inst✝ : MetricSpace Y
                              Φ : Z → X
                              Ψ : Z → Y
                              ε : Real
                              ε0 : LT.lt 0 ε
                              x : Y
                              y : X
                              h : Eq (Metric.glueDist Φ Ψ ε (Sum.inr x) (Sum.inl y)) 0
                              ⊢ Eq (Sum.inr x) (Sum.inl y)
                            -/
  | .inr x, .inl y, h => by exfalso; linarith [le_glueDist_inr_inl Φ Ψ ε x y]
                                     /-
                                       🎉 no goals
                                     -/
                            /-
                              X : Type u
                              Y : Type v
                              Z : Type w
                              inst✝¹ : MetricSpace X
                              inst✝ : MetricSpace Y
                              Φ : Z → X
                              Ψ : Z → Y
                              ε : Real
                              ε0 : LT.lt 0 ε
                              x y : Y
                              h : Eq (Metric.glueDist Φ Ψ ε (Sum.inr x) (Sum.inr y)) 0
                              ⊢ Eq (Sum.inr x) (Sum.inr y)
                            -/
  | .inr x, .inr y, h => by rw [eq_of_dist_eq_zero h]
                            /-
                              🎉 no goals
                            -/


theorem Sum.mem_uniformity_iff_glueDist (hε : 0 < ε) (s : Set ((X ⊕ Y) × (X ⊕ Y))) :
    s ∈ 𝓤 (X ⊕ Y) ↔ ∃ δ > 0, ∀ a b, glueDist Φ Ψ ε a b < δ → (a, b) ∈ s := by
  /-
    X : Type u
    Y : Type v
    Z : Type w
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    Φ : Z → X
    Ψ : Z → Y
    ε : Real
    hε : LT.lt 0 ε
    s : Set (Prod (Sum X Y) (Sum X Y))
    ⊢ Iff (Membership.mem (uniformity (Sum X Y)) s) (Exists fun δ => And (GT.gt δ  …
  -/
  simp only [Sum.uniformity, Filter.mem_sup, Filter.mem_map, mem_uniformity_dist, mem_preimage]
  /-
    X : Type u
    Y : Type v
    Z : Type w
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    Φ : Z → X
    Ψ : Z → Y
    ε : Real
    hε : LT.lt 0 ε
    s : Set (Prod (Sum X Y) (Sum X Y))
    ⊢ Iff (And (Exists fun ε => And (GT.gt ε 0) (∀ ⦃a b : X⦄, LT.lt (Dist.dist a b …
  -/
  constructor
    /-
      case mp
      X : Type u
      Y : Type v
      Z : Type w
      inst✝¹ : MetricSpace X
      inst✝ : MetricSpace Y
      Φ : Z → X
      Ψ : Z → Y
      ε : Real
      hε : LT.lt 0 ε
      s : Set (Prod (Sum X Y) (Sum X Y))
      ⊢ And (Exists fun ε => And (GT.gt ε 0) (∀ ⦃a b : X⦄, LT.lt (Dist.dist a b) ε → …
    -/
  · rintro ⟨⟨δX, δX0, hX⟩, δY, δY0, hY⟩
    /-
      case mp.intro.intro.intro.intro.intro
      X : Type u
      Y : Type v
      Z : Type w
      inst✝¹ : MetricSpace X
      inst✝ : MetricSpace Y
      Φ : Z → X
      Ψ : Z → Y
      ε : Real
      hε : LT.lt 0 ε
      s : Set (Prod (Sum X Y) (Sum X Y))
      δX : Real
      δX0 : GT.gt δX 0
      hX : ∀ ⦃a b : X⦄, LT.lt (Dist.dist a b) δX → Membership.mem s (Prod.map Sum.in …
      δY : Real
      δY0 : GT.gt δY 0
      hY : ∀ ⦃a b : Y⦄, LT.lt (Dist.dist a b) δY → Membership.mem s (Prod.map Sum.in …
      ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (a b : Sum X Y), LT.lt (Metric.glueDist Φ …
    -/
    refine ⟨min (min δX δY) ε, lt_min (lt_min δX0 δY0) hε, ?_⟩
    /-
      case mp.intro.intro.intro.intro.intro
      X : Type u
      Y : Type v
      Z : Type w
      inst✝¹ : MetricSpace X
      inst✝ : MetricSpace Y
      Φ : Z → X
      Ψ : Z → Y
      ε : Real
      hε : LT.lt 0 ε
      s : Set (Prod (Sum X Y) (Sum X Y))
      δX : Real
      δX0 : GT.gt δX 0
      hX : ∀ ⦃a b : X⦄, LT.lt (Dist.dist a b) δX → Membership.mem s (Prod.map Sum.in …
      δY : Real
      δY0 : GT.gt δY 0
      hY : ∀ ⦃a b : Y⦄, LT.lt (Dist.dist a b) δY → Membership.mem s (Prod.map Sum.in …
      ⊢ ∀ (a b : Sum X Y), LT.lt (Metric.glueDist Φ Ψ ε a b) (Min.min (Min.min δX δY …
    -/
    rintro (a | a) (b | b) h <;> simp only [lt_min_iff] at h
      /-
        case mp.intro.intro.intro.intro.intro.inl.inl
        X : Type u
        Y : Type v
        Z : Type w
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        Φ : Z → X
        Ψ : Z → Y
        ε : Real
        hε : LT.lt 0 ε
        s : Set (Prod (Sum X Y) (Sum X Y))
        δX : Real
        δX0 : GT.gt δX 0
        hX : ∀ ⦃a b : X⦄, LT.lt (Dist.dist a b) δX → Membership.mem s (Prod.map Sum.in …
        δY : Real
        δY0 : GT.gt δY 0
        hY : ∀ ⦃a b : Y⦄, LT.lt (Dist.dist a b) δY → Membership.mem s (Prod.map Sum.in …
        a b : X
        h : And (And (LT.lt (Metric.glueDist Φ Ψ ε (Sum.inl a) (Sum.inl b)) δX) (LT.lt …
        ⊢ Membership.mem s { fst := Sum.inl a, snd := Sum.inl b }
      -/
    · exact hX h.1.1
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.intro.intro.inl.inr
        X : Type u
        Y : Type v
        Z : Type w
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        Φ : Z → X
        Ψ : Z → Y
        ε : Real
        hε : LT.lt 0 ε
        s : Set (Prod (Sum X Y) (Sum X Y))
        δX : Real
        δX0 : GT.gt δX 0
        hX : ∀ ⦃a b : X⦄, LT.lt (Dist.dist a b) δX → Membership.mem s (Prod.map Sum.in …
        δY : Real
        δY0 : GT.gt δY 0
        hY : ∀ ⦃a b : Y⦄, LT.lt (Dist.dist a b) δY → Membership.mem s (Prod.map Sum.in …
        a : X
        b : Y
        h : And (And (LT.lt (Metric.glueDist Φ Ψ ε (Sum.inl a) (Sum.inr b)) δX) (LT.lt …
        ⊢ Membership.mem s { fst := Sum.inl a, snd := Sum.inr b }
      -/
    · exact absurd h.2 (le_glueDist_inl_inr _ _ _ _ _).not_lt
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.intro.intro.inr.inl
        X : Type u
        Y : Type v
        Z : Type w
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        Φ : Z → X
        Ψ : Z → Y
        ε : Real
        hε : LT.lt 0 ε
        s : Set (Prod (Sum X Y) (Sum X Y))
        δX : Real
        δX0 : GT.gt δX 0
        hX : ∀ ⦃a b : X⦄, LT.lt (Dist.dist a b) δX → Membership.mem s (Prod.map Sum.in …
        δY : Real
        δY0 : GT.gt δY 0
        hY : ∀ ⦃a b : Y⦄, LT.lt (Dist.dist a b) δY → Membership.mem s (Prod.map Sum.in …
        a : Y
        b : X
        h : And (And (LT.lt (Metric.glueDist Φ Ψ ε (Sum.inr a) (Sum.inl b)) δX) (LT.lt …
        ⊢ Membership.mem s { fst := Sum.inr a, snd := Sum.inl b }
      -/
    · exact absurd h.2 (le_glueDist_inr_inl _ _ _ _ _).not_lt
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.intro.intro.inr.inr
        X : Type u
        Y : Type v
        Z : Type w
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        Φ : Z → X
        Ψ : Z → Y
        ε : Real
        hε : LT.lt 0 ε
        s : Set (Prod (Sum X Y) (Sum X Y))
        δX : Real
        δX0 : GT.gt δX 0
        hX : ∀ ⦃a b : X⦄, LT.lt (Dist.dist a b) δX → Membership.mem s (Prod.map Sum.in …
        δY : Real
        δY0 : GT.gt δY 0
        hY : ∀ ⦃a b : Y⦄, LT.lt (Dist.dist a b) δY → Membership.mem s (Prod.map Sum.in …
        a b : Y
        h : And (And (LT.lt (Metric.glueDist Φ Ψ ε (Sum.inr a) (Sum.inr b)) δX) (LT.lt …
        ⊢ Membership.mem s { fst := Sum.inr a, snd := Sum.inr b }
      -/
    · exact hY h.1.2
      /-
        🎉 no goals
      -/
    /-
      case mpr
      X : Type u
      Y : Type v
      Z : Type w
      inst✝¹ : MetricSpace X
      inst✝ : MetricSpace Y
      Φ : Z → X
      Ψ : Z → Y
      ε : Real
      hε : LT.lt 0 ε
      s : Set (Prod (Sum X Y) (Sum X Y))
      ⊢ (Exists fun δ => And (GT.gt δ 0) (∀ (a b : Sum X Y), LT.lt (Metric.glueDist  …
    -/
  · rintro ⟨ε, ε0, H⟩
    /-
      case mpr.intro.intro
      X : Type u
      Y : Type v
      Z : Type w
      inst✝¹ : MetricSpace X
      inst✝ : MetricSpace Y
      Φ : Z → X
      Ψ : Z → Y
      ε✝ : Real
      hε : LT.lt 0 ε✝
      s : Set (Prod (Sum X Y) (Sum X Y))
      ε : Real
      ε0 : GT.gt ε 0
      H : ∀ (a b : Sum X Y), LT.lt (Metric.glueDist Φ Ψ ε✝ a b) ε → Membership.mem s …
      ⊢ And (Exists fun ε => And (GT.gt ε 0) (∀ ⦃a b : X⦄, LT.lt (Dist.dist a b) ε → …
    -/
                    /-
                      🎉 no goals
                    -/
    constructor <;> exact ⟨ε, ε0, fun _ _ h => H _ _ h⟩
                    /-
                      🎉 no goals
                    -/


/-- Given two maps `Φ` and `Ψ` intro metric spaces `X` and `Y` such that the distances between
`Φ p` and `Φ q`, and between `Ψ p` and `Ψ q`, coincide up to `2 ε` where `ε > 0`, one can almost
glue the two spaces `X` and `Y` along the images of `Φ` and `Ψ`, so that `Φ p` and `Ψ p` are
at distance `ε`. -/
def glueMetricApprox [Nonempty Z] (Φ : Z → X) (Ψ : Z → Y) (ε : ℝ) (ε0 : 0 < ε)
    (H : ∀ p q, |dist (Φ p) (Φ q) - dist (Ψ p) (Ψ q)| ≤ 2 * ε) : MetricSpace (X ⊕ Y) where
  dist := glueDist Φ Ψ ε
  dist_self := glueDist_self Φ Ψ ε
  dist_comm := glueDist_comm Φ Ψ ε
  dist_triangle := glueDist_triangle Φ Ψ ε H
  eq_of_dist_eq_zero := eq_of_glueDist_eq_zero Φ Ψ ε ε0 _ _
  toUniformSpace := Sum.instUniformSpace
  uniformity_dist := uniformity_dist_of_mem_uniformity _ _ <| Sum.mem_uniformity_iff_glueDist ε0


/-- Distance on a disjoint union. There are many (noncanonical) ways to put a distance compatible
with each factor.
If the two spaces are bounded, one can say for instance that each point in the first is at distance
`diam X + diam Y + 1` of each point in the second.
Instead, we choose a construction that works for unbounded spaces, but requires basepoints,
chosen arbitrarily.
We embed isometrically each factor, set the basepoints at distance 1,
arbitrarily, and say that the distance from `a` to `b` is the sum of the distances of `a` and `b` to
their respective basepoints, plus the distance 1 between the basepoints.
Since there is an arbitrary choice in this construction, it is not an instance by default. -/
protected def Sum.dist : X ⊕ Y → X ⊕ Y → ℝ
  | .inl a, .inl a' => dist a a'
  | .inr b, .inr b' => dist b b'
  | .inl a, .inr b => dist a (Nonempty.some ⟨a⟩) + 1 + dist (Nonempty.some ⟨b⟩) b
  | .inr b, .inl a => dist b (Nonempty.some ⟨b⟩) + 1 + dist (Nonempty.some ⟨a⟩) a


theorem Sum.dist_eq_glueDist {p q : X ⊕ Y} (x : X) (y : Y) :
    Sum.dist p q =
      glueDist (fun _ : Unit => Nonempty.some ⟨x⟩) (fun _ : Unit => Nonempty.some ⟨y⟩) 1 p q := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    p q : Sum X Y
    x : X
    y : Y
    ⊢ Eq (Metric.Sum.dist p q) (Metric.glueDist (fun x_1 => ⋯.some) (fun x => ⋯.so …
  -/
  cases p <;> cases q <;> first |rfl|simp [Sum.dist, glueDist, dist_comm, add_comm,
    add_left_comm, add_assoc]


private theorem Sum.dist_comm (x y : X ⊕ Y) : Sum.dist x y = Sum.dist y x := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    x y : Sum X Y
    ⊢ Eq (Metric.Sum.dist x y) (Metric.Sum.dist y x)
  -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
  cases x <;> cases y <;> simp [Sum.dist, _root_.dist_comm, add_comm, add_left_comm, add_assoc]
                          /-
                            🎉 no goals
                          -/


theorem Sum.one_le_dist_inl_inr {x : X} {y : Y} : 1 ≤ Sum.dist (.inl x) (.inr y) :=
  le_trans (le_add_of_nonneg_right dist_nonneg) <|
    add_le_add_right (le_add_of_nonneg_left dist_nonneg) _


theorem Sum.one_le_dist_inr_inl {x : X} {y : Y} : 1 ≤ Sum.dist (.inr y) (.inl x) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    x : X
    y : Y
    ⊢ LE.le 1 (Metric.Sum.dist (Sum.inr y) (Sum.inl x))
  -/
  rw [Sum.dist_comm]; exact Sum.one_le_dist_inl_inr
                      /-
                        🎉 no goals
                      -/


private theorem Sum.mem_uniformity (s : Set ((X ⊕ Y) × (X ⊕ Y))) :
    s ∈ 𝓤 (X ⊕ Y) ↔ ∃ ε > 0, ∀ a b, Sum.dist a b < ε → (a, b) ∈ s := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    s : Set (Prod (Sum X Y) (Sum X Y))
    ⊢ Iff (Membership.mem (uniformity (Sum X Y)) s) (Exists fun ε => And (GT.gt ε  …
  -/
  constructor
    /-
      case mp
      X : Type u
      Y : Type v
      inst✝¹ : MetricSpace X
      inst✝ : MetricSpace Y
      s : Set (Prod (Sum X Y) (Sum X Y))
      ⊢ Membership.mem (uniformity (Sum X Y)) s → Exists fun ε => And (GT.gt ε 0) (∀ …
    -/
  · rintro ⟨hsX, hsY⟩
    /-
      case mp.intro
      X : Type u
      Y : Type v
      inst✝¹ : MetricSpace X
      inst✝ : MetricSpace Y
      s : Set (Prod (Sum X Y) (Sum X Y))
      hsX : Membership.mem (Filter.map (fun p => { fst := Sum.inl p.1, snd := Sum.in …
      hsY : Membership.mem (Filter.map (fun p => { fst := Sum.inr p.1, snd := Sum.in …
      ⊢ Exists fun ε => And (GT.gt ε 0) (∀ (a b : Sum X Y), LT.lt (Metric.Sum.dist a …
    -/
    rcases mem_uniformity_dist.1 hsX with ⟨εX, εX0, hX⟩
    /-
      case mp.intro.intro.intro
      X : Type u
      Y : Type v
      inst✝¹ : MetricSpace X
      inst✝ : MetricSpace Y
      s : Set (Prod (Sum X Y) (Sum X Y))
      hsX : Membership.mem (Filter.map (fun p => { fst := Sum.inl p.1, snd := Sum.in …
      hsY : Membership.mem (Filter.map (fun p => { fst := Sum.inr p.1, snd := Sum.in …
      εX : Real
      εX0 : GT.gt εX 0
      hX : ∀ ⦃a b : X⦄, LT.lt (Dist.dist a b) εX → Membership.mem (Set.preimage (fun …
      ⊢ Exists fun ε => And (GT.gt ε 0) (∀ (a b : Sum X Y), LT.lt (Metric.Sum.dist a …
    -/
    rcases mem_uniformity_dist.1 hsY with ⟨εY, εY0, hY⟩
    /-
      case mp.intro.intro.intro.intro.intro
      X : Type u
      Y : Type v
      inst✝¹ : MetricSpace X
      inst✝ : MetricSpace Y
      s : Set (Prod (Sum X Y) (Sum X Y))
      hsX : Membership.mem (Filter.map (fun p => { fst := Sum.inl p.1, snd := Sum.in …
      hsY : Membership.mem (Filter.map (fun p => { fst := Sum.inr p.1, snd := Sum.in …
      εX : Real
      εX0 : GT.gt εX 0
      hX : ∀ ⦃a b : X⦄, LT.lt (Dist.dist a b) εX → Membership.mem (Set.preimage (fun …
      εY : Real
      εY0 : GT.gt εY 0
      hY : ∀ ⦃a b : Y⦄, LT.lt (Dist.dist a b) εY → Membership.mem (Set.preimage (fun …
      ⊢ Exists fun ε => And (GT.gt ε 0) (∀ (a b : Sum X Y), LT.lt (Metric.Sum.dist a …
    -/
    refine ⟨min (min εX εY) 1, lt_min (lt_min εX0 εY0) zero_lt_one, ?_⟩
    /-
      case mp.intro.intro.intro.intro.intro
      X : Type u
      Y : Type v
      inst✝¹ : MetricSpace X
      inst✝ : MetricSpace Y
      s : Set (Prod (Sum X Y) (Sum X Y))
      hsX : Membership.mem (Filter.map (fun p => { fst := Sum.inl p.1, snd := Sum.in …
      hsY : Membership.mem (Filter.map (fun p => { fst := Sum.inr p.1, snd := Sum.in …
      εX : Real
      εX0 : GT.gt εX 0
      hX : ∀ ⦃a b : X⦄, LT.lt (Dist.dist a b) εX → Membership.mem (Set.preimage (fun …
      εY : Real
      εY0 : GT.gt εY 0
      hY : ∀ ⦃a b : Y⦄, LT.lt (Dist.dist a b) εY → Membership.mem (Set.preimage (fun …
      ⊢ ∀ (a b : Sum X Y), LT.lt (Metric.Sum.dist a b) (Min.min (Min.min εX εY) 1) → …
    -/
    rintro (a | a) (b | b) h
      /-
        case mp.intro.intro.intro.intro.intro.inl.inl
        X : Type u
        Y : Type v
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        s : Set (Prod (Sum X Y) (Sum X Y))
        hsX : Membership.mem (Filter.map (fun p => { fst := Sum.inl p.1, snd := Sum.in …
        hsY : Membership.mem (Filter.map (fun p => { fst := Sum.inr p.1, snd := Sum.in …
        εX : Real
        εX0 : GT.gt εX 0
        hX : ∀ ⦃a b : X⦄, LT.lt (Dist.dist a b) εX → Membership.mem (Set.preimage (fun …
        εY : Real
        εY0 : GT.gt εY 0
        hY : ∀ ⦃a b : Y⦄, LT.lt (Dist.dist a b) εY → Membership.mem (Set.preimage (fun …
        a b : X
        h : LT.lt (Metric.Sum.dist (Sum.inl a) (Sum.inl b)) (Min.min (Min.min εX εY) 1)
        ⊢ Membership.mem s { fst := Sum.inl a, snd := Sum.inl b }
      -/
    · exact hX (lt_of_lt_of_le h (le_trans (min_le_left _ _) (min_le_left _ _)))
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.intro.intro.inl.inr
        X : Type u
        Y : Type v
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        s : Set (Prod (Sum X Y) (Sum X Y))
        hsX : Membership.mem (Filter.map (fun p => { fst := Sum.inl p.1, snd := Sum.in …
        hsY : Membership.mem (Filter.map (fun p => { fst := Sum.inr p.1, snd := Sum.in …
        εX : Real
        εX0 : GT.gt εX 0
        hX : ∀ ⦃a b : X⦄, LT.lt (Dist.dist a b) εX → Membership.mem (Set.preimage (fun …
        εY : Real
        εY0 : GT.gt εY 0
        hY : ∀ ⦃a b : Y⦄, LT.lt (Dist.dist a b) εY → Membership.mem (Set.preimage (fun …
        a : X
        b : Y
        h : LT.lt (Metric.Sum.dist (Sum.inl a) (Sum.inr b)) (Min.min (Min.min εX εY) 1)
        ⊢ Membership.mem s { fst := Sum.inl a, snd := Sum.inr b }
      -/
    · cases not_le_of_lt (lt_of_lt_of_le h (min_le_right _ _)) Sum.one_le_dist_inl_inr
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.intro.intro.inr.inl
        X : Type u
        Y : Type v
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        s : Set (Prod (Sum X Y) (Sum X Y))
        hsX : Membership.mem (Filter.map (fun p => { fst := Sum.inl p.1, snd := Sum.in …
        hsY : Membership.mem (Filter.map (fun p => { fst := Sum.inr p.1, snd := Sum.in …
        εX : Real
        εX0 : GT.gt εX 0
        hX : ∀ ⦃a b : X⦄, LT.lt (Dist.dist a b) εX → Membership.mem (Set.preimage (fun …
        εY : Real
        εY0 : GT.gt εY 0
        hY : ∀ ⦃a b : Y⦄, LT.lt (Dist.dist a b) εY → Membership.mem (Set.preimage (fun …
        a : Y
        b : X
        h : LT.lt (Metric.Sum.dist (Sum.inr a) (Sum.inl b)) (Min.min (Min.min εX εY) 1)
        ⊢ Membership.mem s { fst := Sum.inr a, snd := Sum.inl b }
      -/
    · cases not_le_of_lt (lt_of_lt_of_le h (min_le_right _ _)) Sum.one_le_dist_inr_inl
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.intro.intro.inr.inr
        X : Type u
        Y : Type v
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        s : Set (Prod (Sum X Y) (Sum X Y))
        hsX : Membership.mem (Filter.map (fun p => { fst := Sum.inl p.1, snd := Sum.in …
        hsY : Membership.mem (Filter.map (fun p => { fst := Sum.inr p.1, snd := Sum.in …
        εX : Real
        εX0 : GT.gt εX 0
        hX : ∀ ⦃a b : X⦄, LT.lt (Dist.dist a b) εX → Membership.mem (Set.preimage (fun …
        εY : Real
        εY0 : GT.gt εY 0
        hY : ∀ ⦃a b : Y⦄, LT.lt (Dist.dist a b) εY → Membership.mem (Set.preimage (fun …
        a b : Y
        h : LT.lt (Metric.Sum.dist (Sum.inr a) (Sum.inr b)) (Min.min (Min.min εX εY) 1)
        ⊢ Membership.mem s { fst := Sum.inr a, snd := Sum.inr b }
      -/
    · exact hY (lt_of_lt_of_le h (le_trans (min_le_left _ _) (min_le_right _ _)))
      /-
        🎉 no goals
      -/
    /-
      case mpr
      X : Type u
      Y : Type v
      inst✝¹ : MetricSpace X
      inst✝ : MetricSpace Y
      s : Set (Prod (Sum X Y) (Sum X Y))
      ⊢ (Exists fun ε => And (GT.gt ε 0) (∀ (a b : Sum X Y), LT.lt (Metric.Sum.dist  …
    -/
  · rintro ⟨ε, ε0, H⟩
    /-
      case mpr.intro.intro
      X : Type u
      Y : Type v
      inst✝¹ : MetricSpace X
      inst✝ : MetricSpace Y
      s : Set (Prod (Sum X Y) (Sum X Y))
      ε : Real
      ε0 : GT.gt ε 0
      H : ∀ (a b : Sum X Y), LT.lt (Metric.Sum.dist a b) ε → Membership.mem s { fst  …
      ⊢ Membership.mem (uniformity (Sum X Y)) s
    -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
    constructor <;> rw [Filter.mem_map, mem_uniformity_dist] <;> exact ⟨ε, ε0, fun _ _ h => H _ _ h⟩
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- The distance on the disjoint union indeed defines a metric space. All the distance properties
follow from our choice of the distance. The harder work is to show that the uniform structure
defined by the distance coincides with the disjoint union uniform structure. -/
def metricSpaceSum : MetricSpace (X ⊕ Y) where
  dist := Sum.dist
                    /-
                      X : Type u
                      Y : Type v
                      Z : Type w
                      inst✝¹ : MetricSpace X
                      inst✝ : MetricSpace Y
                      x : Sum X Y
                      ⊢ Eq (Dist.dist x x) 0
                    -/
                                /-
                                  🎉 no goals
                                -/
  dist_self x := by cases x <;> simp only [Sum.dist, dist_self]
                                /-
                                  🎉 no goals
                                -/
  dist_comm := Sum.dist_comm
  dist_triangle
    | .inl p, .inl q, .inl r => dist_triangle p q r
    | .inl p, .inr q, _ => by
      /-
        X : Type u
        Y : Type v
        Z : Type w
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        p : X
        q : Y
        x✝ : Sum X Y
        ⊢ LE.le (Dist.dist (Sum.inl p) x✝) (HAdd.hAdd (Dist.dist (Sum.inl p) (Sum.inr  …
      -/
      simp only [Sum.dist_eq_glueDist p q]
      /-
        X : Type u
        Y : Type v
        Z : Type w
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        p : X
        q : Y
        x✝ : Sum X Y
        ⊢ LE.le (Metric.glueDist (fun x => ⋯.some) (fun x => ⋯.some) 1 (Sum.inl p) x✝) …
      -/
      exact glueDist_triangle _ _ _ (by norm_num) _ _ _
      /-
        🎉 no goals
      -/
    | _, .inl q, .inr r => by
      /-
        X : Type u
        Y : Type v
        Z : Type w
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        x✝ : Sum X Y
        q : X
        r : Y
        ⊢ LE.le (Dist.dist x✝ (Sum.inr r)) (HAdd.hAdd (Dist.dist x✝ (Sum.inl q)) (Dist …
      -/
      simp only [Sum.dist_eq_glueDist q r]
      /-
        X : Type u
        Y : Type v
        Z : Type w
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        x✝ : Sum X Y
        q : X
        r : Y
        ⊢ LE.le (Metric.glueDist (fun x => ⋯.some) (fun x => ⋯.some) 1 x✝ (Sum.inr r)) …
      -/
      exact glueDist_triangle _ _ _ (by norm_num) _ _ _
      /-
        🎉 no goals
      -/
    | .inr p, _, .inl r => by
      /-
        X : Type u
        Y : Type v
        Z : Type w
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        p : Y
        x✝ : Sum X Y
        r : X
        ⊢ LE.le (Dist.dist (Sum.inr p) (Sum.inl r)) (HAdd.hAdd (Dist.dist (Sum.inr p)  …
      -/
      simp only [Sum.dist_eq_glueDist r p]
      /-
        X : Type u
        Y : Type v
        Z : Type w
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        p : Y
        x✝ : Sum X Y
        r : X
        ⊢ LE.le (Metric.glueDist (fun x => ⋯.some) (fun x => ⋯.some) 1 (Sum.inr p) (Su …
      -/
      exact glueDist_triangle _ _ _ (by norm_num) _ _ _
      /-
        🎉 no goals
      -/
    | .inr p, .inr q, .inr r => dist_triangle p q r
  eq_of_dist_eq_zero {p q} h := by
    /-
      X : Type u
      Y : Type v
      Z : Type w
      inst✝¹ : MetricSpace X
      inst✝ : MetricSpace Y
      p q : Sum X Y
      h : Eq (Dist.dist p q) 0
      ⊢ Eq p q
    -/
    cases' p with p p <;> cases' q with q q
      /-
        case inl.inl
        X : Type u
        Y : Type v
        Z : Type w
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        p q : X
        h : Eq (Dist.dist (Sum.inl p) (Sum.inl q)) 0
        ⊢ Eq (Sum.inl p) (Sum.inl q)
      -/
    · rw [eq_of_dist_eq_zero h]
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        X : Type u
        Y : Type v
        Z : Type w
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        p : X
        q : Y
        h : Eq (Dist.dist (Sum.inl p) (Sum.inr q)) 0
        ⊢ Eq (Sum.inl p) (Sum.inr q)
      -/
    · exact eq_of_glueDist_eq_zero _ _ _ one_pos _ _ ((Sum.dist_eq_glueDist p q).symm.trans h)
      /-
        🎉 no goals
      -/
      /-
        case inr.inl
        X : Type u
        Y : Type v
        Z : Type w
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        p : Y
        q : X
        h : Eq (Dist.dist (Sum.inr p) (Sum.inl q)) 0
        ⊢ Eq (Sum.inr p) (Sum.inl q)
      -/
    · exact eq_of_glueDist_eq_zero _ _ _ one_pos _ _ ((Sum.dist_eq_glueDist q p).symm.trans h)
      /-
        🎉 no goals
      -/
      /-
        case inr.inr
        X : Type u
        Y : Type v
        Z : Type w
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        p q : Y
        h : Eq (Dist.dist (Sum.inr p) (Sum.inr q)) 0
        ⊢ Eq (Sum.inr p) (Sum.inr q)
      -/
    · rw [eq_of_dist_eq_zero h]
      /-
        🎉 no goals
      -/
  toUniformSpace := Sum.instUniformSpace
  uniformity_dist := uniformity_dist_of_mem_uniformity _ _ Sum.mem_uniformity


theorem Sum.dist_eq {x y : X ⊕ Y} : dist x y = Sum.dist x y := rfl


/-- The left injection of a space in a disjoint union is an isometry -/
theorem isometry_inl : Isometry (Sum.inl : X → X ⊕ Y) :=
  Isometry.of_dist_eq fun _ _ => rfl


/-- The right injection of a space in a disjoint union is an isometry -/
theorem isometry_inr : Isometry (Sum.inr : Y → X ⊕ Y) :=
  Isometry.of_dist_eq fun _ _ => rfl


open Classical in
/-- Distance on a disjoint union. There are many (noncanonical) ways to put a distance compatible
with each factor.
We choose a construction that works for unbounded spaces, but requires basepoints,
chosen arbitrarily.
We embed isometrically each factor, set the basepoints at distance 1, arbitrarily,
and say that the distance from `a` to `b` is the sum of the distances of `a` and `b` to
their respective basepoints, plus the distance 1 between the basepoints.
Since there is an arbitrary choice in this construction, it is not an instance by default. -/
protected def dist : (Σ i, E i) → (Σ i, E i) → ℝ
  | ⟨i, x⟩, ⟨j, y⟩ =>
    if h : i = j then
                              /-
                                ι : Type u_1
                                E : ι → Type u_2
                                inst✝ : (i : ι) → MetricSpace (E i)
                                i : ι
                                x : E i
                                j : ι
                                y : E j
                                h : Eq i j
                                ⊢ Eq (E j) (E i)
                              -/
      haveI : E j = E i := by rw [h]
                              /-
                                🎉 no goals
                              -/
      Dist.dist x (cast this y)
    else Dist.dist x (Nonempty.some ⟨x⟩) + 1 + Dist.dist (Nonempty.some ⟨y⟩) y


/-- A `Dist` instance on the disjoint union `Σ i, E i`.
We embed isometrically each factor, set the basepoints at distance 1, arbitrarily,
and say that the distance from `a` to `b` is the sum of the distances of `a` and `b` to
their respective basepoints, plus the distance 1 between the basepoints.
Since there is an arbitrary choice in this construction, it is not an instance by default. -/
def instDist : Dist (Σi, E i) :=
  ⟨Sigma.dist⟩


@[simp]
theorem dist_same (i : ι) (x y : E i) : dist (Sigma.mk i x) ⟨i, y⟩ = dist x y := by
  /-
    ι : Type u_1
    E : ι → Type u_2
    inst✝ : (i : ι) → MetricSpace (E i)
    i : ι
    x y : E i
    ⊢ Eq (Dist.dist ⟨i, x⟩ ⟨i, y⟩) (Dist.dist x y)
  -/
  simp [Dist.dist, Sigma.dist]
  /-
    🎉 no goals
  -/


@[simp]
theorem dist_ne {i j : ι} (h : i ≠ j) (x : E i) (y : E j) :
    dist (⟨i, x⟩ : Σk, E k) ⟨j, y⟩ = dist x (Nonempty.some ⟨x⟩) + 1 + dist (Nonempty.some ⟨y⟩) y :=
  dif_neg h


theorem one_le_dist_of_ne {i j : ι} (h : i ≠ j) (x : E i) (y : E j) :
    1 ≤ dist (⟨i, x⟩ : Σk, E k) ⟨j, y⟩ := by
  /-
    ι : Type u_1
    E : ι → Type u_2
    inst✝ : (i : ι) → MetricSpace (E i)
    i j : ι
    h : Ne i j
    x : E i
    y : E j
    ⊢ LE.le 1 (Dist.dist ⟨i, x⟩ ⟨j, y⟩)
  -/
  rw [Sigma.dist_ne h x y]
  /-
    ι : Type u_1
    E : ι → Type u_2
    inst✝ : (i : ι) → MetricSpace (E i)
    i j : ι
    h : Ne i j
    x : E i
    y : E j
    ⊢ LE.le 1 (HAdd.hAdd (HAdd.hAdd (Dist.dist x ⋯.some) 1) (Dist.dist ⋯.some y))
  -/
  linarith [@dist_nonneg _ _ x (Nonempty.some ⟨x⟩), @dist_nonneg _ _ (Nonempty.some ⟨y⟩) y]
  /-
    🎉 no goals
  -/


theorem fst_eq_of_dist_lt_one (x y : Σi, E i) (h : dist x y < 1) : x.1 = y.1 := by
  /-
    ι : Type u_1
    E : ι → Type u_2
    inst✝ : (i : ι) → MetricSpace (E i)
    x y : Sigma fun i => E i
    h : LT.lt (Dist.dist x y) 1
    ⊢ Eq x.fst y.fst
  -/
  cases x; cases y
  /-
    case mk.mk
    ι : Type u_1
    E : ι → Type u_2
    inst✝ : (i : ι) → MetricSpace (E i)
    fst✝¹ : ι
    snd✝¹ : E fst✝¹
    fst✝ : ι
    snd✝ : E fst✝
    h : LT.lt (Dist.dist ⟨fst✝¹, snd✝¹⟩ ⟨fst✝, snd✝⟩) 1
    ⊢ Eq ⟨fst✝¹, snd✝¹⟩.fst ⟨fst✝, snd✝⟩.fst
  -/
  contrapose! h
  /-
    case mk.mk
    ι : Type u_1
    E : ι → Type u_2
    inst✝ : (i : ι) → MetricSpace (E i)
    fst✝¹ : ι
    snd✝¹ : E fst✝¹
    fst✝ : ι
    snd✝ : E fst✝
    h : Ne ⟨fst✝¹, snd✝¹⟩.fst ⟨fst✝, snd✝⟩.fst
    ⊢ LE.le 1 (Dist.dist ⟨fst✝¹, snd✝¹⟩ ⟨fst✝, snd✝⟩)
  -/
  apply one_le_dist_of_ne h
  /-
    🎉 no goals
  -/


protected theorem dist_triangle (x y z : Σi, E i) : dist x z ≤ dist x y + dist y z := by
  /-
    ι : Type u_1
    E : ι → Type u_2
    inst✝ : (i : ι) → MetricSpace (E i)
    x y z : Sigma fun i => E i
    ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
  -/
  rcases x with ⟨i, x⟩; rcases y with ⟨j, y⟩; rcases z with ⟨k, z⟩
  /-
    case mk.mk.mk
    ι : Type u_1
    E : ι → Type u_2
    inst✝ : (i : ι) → MetricSpace (E i)
    i : ι
    x : E i
    j : ι
    y : E j
    k : ι
    z : E k
    ⊢ LE.le (Dist.dist ⟨i, x⟩ ⟨k, z⟩) (HAdd.hAdd (Dist.dist ⟨i, x⟩ ⟨j, y⟩) (Dist.d …
  -/
  rcases eq_or_ne i k with (rfl | hik)
    /-
      case mk.mk.mk.inl
      ι : Type u_1
      E : ι → Type u_2
      inst✝ : (i : ι) → MetricSpace (E i)
      i : ι
      x : E i
      j : ι
      y : E j
      z : E i
      ⊢ LE.le (Dist.dist ⟨i, x⟩ ⟨i, z⟩) (HAdd.hAdd (Dist.dist ⟨i, x⟩ ⟨j, y⟩) (Dist.d …
    -/
  · rcases eq_or_ne i j with (rfl | hij)
      /-
        case mk.mk.mk.inl.inl
        ι : Type u_1
        E : ι → Type u_2
        inst✝ : (i : ι) → MetricSpace (E i)
        i : ι
        x z y : E i
        ⊢ LE.le (Dist.dist ⟨i, x⟩ ⟨i, z⟩) (HAdd.hAdd (Dist.dist ⟨i, x⟩ ⟨i, y⟩) (Dist.d …
      -/
    · simpa using dist_triangle x y z
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.mk.inl.inr
        ι : Type u_1
        E : ι → Type u_2
        inst✝ : (i : ι) → MetricSpace (E i)
        i : ι
        x : E i
        j : ι
        y : E j
        z : E i
        hij : Ne i j
        ⊢ LE.le (Dist.dist ⟨i, x⟩ ⟨i, z⟩) (HAdd.hAdd (Dist.dist ⟨i, x⟩ ⟨j, y⟩) (Dist.d …
      -/
    · simp only [Sigma.dist_same, Sigma.dist_ne hij, Sigma.dist_ne hij.symm]
      calc
        dist x z ≤ dist x (Nonempty.some ⟨x⟩) + 0 + 0 + (0 + 0 + dist (Nonempty.some ⟨z⟩) z) := by
          simpa only [zero_add, add_zero] using dist_triangle _ _ _
        _ ≤ _ := by apply_rules [add_le_add, le_rfl, dist_nonneg, zero_le_one]
    /-
      case mk.mk.mk.inr
      ι : Type u_1
      E : ι → Type u_2
      inst✝ : (i : ι) → MetricSpace (E i)
      i : ι
      x : E i
      j : ι
      y : E j
      k : ι
      z : E k
      hik : Ne i k
      ⊢ LE.le (Dist.dist ⟨i, x⟩ ⟨k, z⟩) (HAdd.hAdd (Dist.dist ⟨i, x⟩ ⟨j, y⟩) (Dist.d …
    -/
  · rcases eq_or_ne i j with (rfl | hij)
      /-
        case mk.mk.mk.inr.inl
        ι : Type u_1
        E : ι → Type u_2
        inst✝ : (i : ι) → MetricSpace (E i)
        i : ι
        x : E i
        k : ι
        z : E k
        hik : Ne i k
        y : E i
        ⊢ LE.le (Dist.dist ⟨i, x⟩ ⟨k, z⟩) (HAdd.hAdd (Dist.dist ⟨i, x⟩ ⟨i, y⟩) (Dist.d …
      -/
    · simp only [Sigma.dist_ne hik, Sigma.dist_same]
      calc
        dist x (Nonempty.some ⟨x⟩) + 1 + dist (Nonempty.some ⟨z⟩) z ≤
            dist x y + dist y (Nonempty.some ⟨y⟩) + 1 + dist (Nonempty.some ⟨z⟩) z := by
          apply_rules [add_le_add, le_rfl, dist_triangle]
        _ = _ := by abel
      /-
        case mk.mk.mk.inr.inr
        ι : Type u_1
        E : ι → Type u_2
        inst✝ : (i : ι) → MetricSpace (E i)
        i : ι
        x : E i
        j : ι
        y : E j
        k : ι
        z : E k
        hik : Ne i k
        hij : Ne i j
        ⊢ LE.le (Dist.dist ⟨i, x⟩ ⟨k, z⟩) (HAdd.hAdd (Dist.dist ⟨i, x⟩ ⟨j, y⟩) (Dist.d …
      -/
    · rcases eq_or_ne j k with (rfl | hjk)
        /-
          case mk.mk.mk.inr.inr.inl
          ι : Type u_1
          E : ι → Type u_2
          inst✝ : (i : ι) → MetricSpace (E i)
          i : ι
          x : E i
          j : ι
          y : E j
          hij : Ne i j
          z : E j
          hik : Ne i j
          ⊢ LE.le (Dist.dist ⟨i, x⟩ ⟨j, z⟩) (HAdd.hAdd (Dist.dist ⟨i, x⟩ ⟨j, y⟩) (Dist.d …
        -/
      · simp only [Sigma.dist_ne hij, Sigma.dist_same]
        calc
          dist x (Nonempty.some ⟨x⟩) + 1 + dist (Nonempty.some ⟨z⟩) z ≤
              dist x (Nonempty.some ⟨x⟩) + 1 + (dist (Nonempty.some ⟨z⟩) y + dist y z) := by
            apply_rules [add_le_add, le_rfl, dist_triangle]
          _ = _ := by abel
        /-
          case mk.mk.mk.inr.inr.inr
          ι : Type u_1
          E : ι → Type u_2
          inst✝ : (i : ι) → MetricSpace (E i)
          i : ι
          x : E i
          j : ι
          y : E j
          k : ι
          z : E k
          hik : Ne i k
          hij : Ne i j
          hjk : Ne j k
          ⊢ LE.le (Dist.dist ⟨i, x⟩ ⟨k, z⟩) (HAdd.hAdd (Dist.dist ⟨i, x⟩ ⟨j, y⟩) (Dist.d …
        -/
      · simp only [hik, hij, hjk, Sigma.dist_ne, Ne, not_false_iff]
        calc
          dist x (Nonempty.some ⟨x⟩) + 1 + dist (Nonempty.some ⟨z⟩) z =
              dist x (Nonempty.some ⟨x⟩) + 1 + 0 + (0 + 0 + dist (Nonempty.some ⟨z⟩) z) := by
            simp only [add_zero, zero_add]
          _ ≤ _ := by apply_rules [add_le_add, zero_le_one, dist_nonneg, le_rfl]


protected theorem isOpen_iff (s : Set (Σi, E i)) :
    IsOpen s ↔ ∀ x ∈ s, ∃ ε > 0, ∀ y, dist x y < ε → y ∈ s := by
  /-
    ι : Type u_1
    E : ι → Type u_2
    inst✝ : (i : ι) → MetricSpace (E i)
    s : Set (Sigma fun i => E i)
    ⊢ Iff (IsOpen s) (∀ (x : Sigma fun i => E i), Membership.mem s x → Exists fun  …
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      E : ι → Type u_2
      inst✝ : (i : ι) → MetricSpace (E i)
      s : Set (Sigma fun i => E i)
      ⊢ IsOpen s → ∀ (x : Sigma fun i => E i), Membership.mem s x → Exists fun ε =>  …
    -/
  · rintro hs ⟨i, x⟩ hx
    obtain ⟨ε, εpos, hε⟩ : ∃ ε > 0, ball x ε ⊆ Sigma.mk i ⁻¹' s :=
      Metric.isOpen_iff.1 (isOpen_sigma_iff.1 hs i) x hx
    /-
      case mp.mk.intro.intro
      ι : Type u_1
      E : ι → Type u_2
      inst✝ : (i : ι) → MetricSpace (E i)
      s : Set (Sigma fun i => E i)
      hs : IsOpen s
      i : ι
      x : E i
      hx : Membership.mem s ⟨i, x⟩
      ε : Real
      εpos : GT.gt ε 0
      hε : HasSubset.Subset (Metric.ball x ε) (Set.preimage (Sigma.mk i) s)
      ⊢ Exists fun ε => And (GT.gt ε 0) (∀ (y : Sigma fun i => E i), LT.lt (Dist.dis …
    -/
    refine ⟨min ε 1, lt_min εpos zero_lt_one, ?_⟩
    /-
      case mp.mk.intro.intro
      ι : Type u_1
      E : ι → Type u_2
      inst✝ : (i : ι) → MetricSpace (E i)
      s : Set (Sigma fun i => E i)
      hs : IsOpen s
      i : ι
      x : E i
      hx : Membership.mem s ⟨i, x⟩
      ε : Real
      εpos : GT.gt ε 0
      hε : HasSubset.Subset (Metric.ball x ε) (Set.preimage (Sigma.mk i) s)
      ⊢ ∀ (y : Sigma fun i => E i), LT.lt (Dist.dist ⟨i, x⟩ y) (Min.min ε 1) → Membe …
    -/
    rintro ⟨j, y⟩ hy
    /-
      case mp.mk.intro.intro.mk
      ι : Type u_1
      E : ι → Type u_2
      inst✝ : (i : ι) → MetricSpace (E i)
      s : Set (Sigma fun i => E i)
      hs : IsOpen s
      i : ι
      x : E i
      hx : Membership.mem s ⟨i, x⟩
      ε : Real
      εpos : GT.gt ε 0
      hε : HasSubset.Subset (Metric.ball x ε) (Set.preimage (Sigma.mk i) s)
      j : ι
      y : E j
      hy : LT.lt (Dist.dist ⟨i, x⟩ ⟨j, y⟩) (Min.min ε 1)
      ⊢ Membership.mem s ⟨j, y⟩
    -/
    rcases eq_or_ne i j with (rfl | hij)
      /-
        case mp.mk.intro.intro.mk.inl
        ι : Type u_1
        E : ι → Type u_2
        inst✝ : (i : ι) → MetricSpace (E i)
        s : Set (Sigma fun i => E i)
        hs : IsOpen s
        i : ι
        x : E i
        hx : Membership.mem s ⟨i, x⟩
        ε : Real
        εpos : GT.gt ε 0
        hε : HasSubset.Subset (Metric.ball x ε) (Set.preimage (Sigma.mk i) s)
        y : E i
        hy : LT.lt (Dist.dist ⟨i, x⟩ ⟨i, y⟩) (Min.min ε 1)
        ⊢ Membership.mem s ⟨i, y⟩
      -/
    · simp only [Sigma.dist_same, lt_min_iff] at hy
      /-
        case mp.mk.intro.intro.mk.inl
        ι : Type u_1
        E : ι → Type u_2
        inst✝ : (i : ι) → MetricSpace (E i)
        s : Set (Sigma fun i => E i)
        hs : IsOpen s
        i : ι
        x : E i
        hx : Membership.mem s ⟨i, x⟩
        ε : Real
        εpos : GT.gt ε 0
        hε : HasSubset.Subset (Metric.ball x ε) (Set.preimage (Sigma.mk i) s)
        y : E i
        hy : And (LT.lt (Dist.dist x y) ε) (LT.lt (Dist.dist x y) 1)
        ⊢ Membership.mem s ⟨i, y⟩
      -/
      exact hε (mem_ball'.2 hy.1)
      /-
        🎉 no goals
      -/
      /-
        case mp.mk.intro.intro.mk.inr
        ι : Type u_1
        E : ι → Type u_2
        inst✝ : (i : ι) → MetricSpace (E i)
        s : Set (Sigma fun i => E i)
        hs : IsOpen s
        i : ι
        x : E i
        hx : Membership.mem s ⟨i, x⟩
        ε : Real
        εpos : GT.gt ε 0
        hε : HasSubset.Subset (Metric.ball x ε) (Set.preimage (Sigma.mk i) s)
        j : ι
        y : E j
        hy : LT.lt (Dist.dist ⟨i, x⟩ ⟨j, y⟩) (Min.min ε 1)
        hij : Ne i j
        ⊢ Membership.mem s ⟨j, y⟩
      -/
    · apply (lt_irrefl (1 : ℝ) _).elim
      calc
        1 ≤ Sigma.dist ⟨i, x⟩ ⟨j, y⟩ := Sigma.one_le_dist_of_ne hij _ _
        _ < 1 := hy.trans_le (min_le_right _ _)
    /-
      case mpr
      ι : Type u_1
      E : ι → Type u_2
      inst✝ : (i : ι) → MetricSpace (E i)
      s : Set (Sigma fun i => E i)
      ⊢ (∀ (x : Sigma fun i => E i), Membership.mem s x → Exists fun ε => And (GT.gt …
    -/
  · refine fun H => isOpen_sigma_iff.2 fun i => Metric.isOpen_iff.2 fun x hx => ?_
    obtain ⟨ε, εpos, hε⟩ : ∃ ε > 0, ∀ y, dist (⟨i, x⟩ : Σj, E j) y < ε → y ∈ s :=
      H ⟨i, x⟩ hx
    /-
      case mpr.intro.intro
      ι : Type u_1
      E : ι → Type u_2
      inst✝ : (i : ι) → MetricSpace (E i)
      s : Set (Sigma fun i => E i)
      H : ∀ (x : Sigma fun i => E i), Membership.mem s x → Exists fun ε => And (GT.g …
      i : ι
      x : E i
      hx : Membership.mem (Set.preimage (Sigma.mk i) s) x
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ (y : Sigma fun j => E j), LT.lt (Dist.dist ⟨i, x⟩ y) ε → Membership.mem …
      ⊢ Exists fun ε => And (GT.gt ε 0) (HasSubset.Subset (Metric.ball x ε) (Set.pre …
    -/
    refine ⟨ε, εpos, fun y hy => ?_⟩
    /-
      case mpr.intro.intro
      ι : Type u_1
      E : ι → Type u_2
      inst✝ : (i : ι) → MetricSpace (E i)
      s : Set (Sigma fun i => E i)
      H : ∀ (x : Sigma fun i => E i), Membership.mem s x → Exists fun ε => And (GT.g …
      i : ι
      x : E i
      hx : Membership.mem (Set.preimage (Sigma.mk i) s) x
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ (y : Sigma fun j => E j), LT.lt (Dist.dist ⟨i, x⟩ y) ε → Membership.mem …
      y : E i
      hy : Membership.mem (Metric.ball x ε) y
      ⊢ Membership.mem (Set.preimage (Sigma.mk i) s) y
    -/
    apply hε ⟨i, y⟩
    /-
      case mpr.intro.intro
      ι : Type u_1
      E : ι → Type u_2
      inst✝ : (i : ι) → MetricSpace (E i)
      s : Set (Sigma fun i => E i)
      H : ∀ (x : Sigma fun i => E i), Membership.mem s x → Exists fun ε => And (GT.g …
      i : ι
      x : E i
      hx : Membership.mem (Set.preimage (Sigma.mk i) s) x
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ (y : Sigma fun j => E j), LT.lt (Dist.dist ⟨i, x⟩ y) ε → Membership.mem …
      y : E i
      hy : Membership.mem (Metric.ball x ε) y
      ⊢ LT.lt (Dist.dist ⟨i, x⟩ ⟨i, y⟩) ε
    -/
    rw [Sigma.dist_same]
    /-
      case mpr.intro.intro
      ι : Type u_1
      E : ι → Type u_2
      inst✝ : (i : ι) → MetricSpace (E i)
      s : Set (Sigma fun i => E i)
      H : ∀ (x : Sigma fun i => E i), Membership.mem s x → Exists fun ε => And (GT.g …
      i : ι
      x : E i
      hx : Membership.mem (Set.preimage (Sigma.mk i) s) x
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ (y : Sigma fun j => E j), LT.lt (Dist.dist ⟨i, x⟩ y) ε → Membership.mem …
      y : E i
      hy : Membership.mem (Metric.ball x ε) y
      ⊢ LT.lt (Dist.dist x y) ε
    -/
    exact mem_ball'.1 hy
    /-
      🎉 no goals
    -/


/-- A metric space structure on the disjoint union `Σ i, E i`.
We embed isometrically each factor, set the basepoints at distance 1, arbitrarily,
and say that the distance from `a` to `b` is the sum of the distances of `a` and `b` to
their respective basepoints, plus the distance 1 between the basepoints.
Since there is an arbitrary choice in this construction, it is not an instance by default. -/
protected def metricSpace : MetricSpace (Σi, E i) := by
  /-
    ι : Type u_1
    E : ι → Type u_2
    inst✝ : (i : ι) → MetricSpace (E i)
    ⊢ MetricSpace (Sigma fun i => E i)
  -/
  refine MetricSpace.ofDistTopology Sigma.dist ?_ ?_ Sigma.dist_triangle Sigma.isOpen_iff ?_
    /-
      case refine_1
      ι : Type u_1
      E : ι → Type u_2
      inst✝ : (i : ι) → MetricSpace (E i)
      ⊢ ∀ (x : Sigma fun i => E i), Eq (Metric.Sigma.dist x x) 0
    -/
  · rintro ⟨i, x⟩
    /-
      case refine_1.mk
      ι : Type u_1
      E : ι → Type u_2
      inst✝ : (i : ι) → MetricSpace (E i)
      i : ι
      x : E i
      ⊢ Eq (Metric.Sigma.dist ⟨i, x⟩ ⟨i, x⟩) 0
    -/
    simp [Sigma.dist]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      E : ι → Type u_2
      inst✝ : (i : ι) → MetricSpace (E i)
      ⊢ ∀ (x y : Sigma fun i => E i), Eq (Metric.Sigma.dist x y) (Metric.Sigma.dist  …
    -/
  · rintro ⟨i, x⟩ ⟨j, y⟩
    /-
      case refine_2.mk.mk
      ι : Type u_1
      E : ι → Type u_2
      inst✝ : (i : ι) → MetricSpace (E i)
      i : ι
      x : E i
      j : ι
      y : E j
      ⊢ Eq (Metric.Sigma.dist ⟨i, x⟩ ⟨j, y⟩) (Metric.Sigma.dist ⟨j, y⟩ ⟨i, x⟩)
    -/
    rcases eq_or_ne i j with (rfl | h)
      /-
        case refine_2.mk.mk.inl
        ι : Type u_1
        E : ι → Type u_2
        inst✝ : (i : ι) → MetricSpace (E i)
        i : ι
        x y : E i
        ⊢ Eq (Metric.Sigma.dist ⟨i, x⟩ ⟨i, y⟩) (Metric.Sigma.dist ⟨i, y⟩ ⟨i, x⟩)
      -/
    · simp [Sigma.dist, dist_comm]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.mk.mk.inr
        ι : Type u_1
        E : ι → Type u_2
        inst✝ : (i : ι) → MetricSpace (E i)
        i : ι
        x : E i
        j : ι
        y : E j
        h : Ne i j
        ⊢ Eq (Metric.Sigma.dist ⟨i, x⟩ ⟨j, y⟩) (Metric.Sigma.dist ⟨j, y⟩ ⟨i, x⟩)
      -/
    · simp only [Sigma.dist, dist_comm, h, h.symm, not_false_iff, dif_neg]
      /-
        case refine_2.mk.mk.inr
        ι : Type u_1
        E : ι → Type u_2
        inst✝ : (i : ι) → MetricSpace (E i)
        i : ι
        x : E i
        j : ι
        y : E j
        h : Ne i j
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Dist.dist x ⋯.some) 1) (Dist.dist y ⋯.some)) (HAdd …
      -/
      /-
        🎉 no goals
      -/
      abel
      /-
        🎉 no goals
      -/
    /-
      case refine_3
      ι : Type u_1
      E : ι → Type u_2
      inst✝ : (i : ι) → MetricSpace (E i)
      ⊢ ∀ (x y : Sigma fun i => E i), Eq (Metric.Sigma.dist x y) 0 → Eq x y
    -/
  · rintro ⟨i, x⟩ ⟨j, y⟩
    /-
      case refine_3.mk.mk
      ι : Type u_1
      E : ι → Type u_2
      inst✝ : (i : ι) → MetricSpace (E i)
      i : ι
      x : E i
      j : ι
      y : E j
      ⊢ Eq (Metric.Sigma.dist ⟨i, x⟩ ⟨j, y⟩) 0 → Eq ⟨i, x⟩ ⟨j, y⟩
    -/
    rcases eq_or_ne i j with (rfl | hij)
      /-
        case refine_3.mk.mk.inl
        ι : Type u_1
        E : ι → Type u_2
        inst✝ : (i : ι) → MetricSpace (E i)
        i : ι
        x y : E i
        ⊢ Eq (Metric.Sigma.dist ⟨i, x⟩ ⟨i, y⟩) 0 → Eq ⟨i, x⟩ ⟨i, y⟩
      -/
    · simp [Sigma.dist]
      /-
        🎉 no goals
      -/
      /-
        case refine_3.mk.mk.inr
        ι : Type u_1
        E : ι → Type u_2
        inst✝ : (i : ι) → MetricSpace (E i)
        i : ι
        x : E i
        j : ι
        y : E j
        hij : Ne i j
        ⊢ Eq (Metric.Sigma.dist ⟨i, x⟩ ⟨j, y⟩) 0 → Eq ⟨i, x⟩ ⟨j, y⟩
      -/
    · intro h
      /-
        case refine_3.mk.mk.inr
        ι : Type u_1
        E : ι → Type u_2
        inst✝ : (i : ι) → MetricSpace (E i)
        i : ι
        x : E i
        j : ι
        y : E j
        hij : Ne i j
        h : Eq (Metric.Sigma.dist ⟨i, x⟩ ⟨j, y⟩) 0
        ⊢ Eq ⟨i, x⟩ ⟨j, y⟩
      -/
      apply (lt_irrefl (1 : ℝ) _).elim
      calc
        1 ≤ Sigma.dist (⟨i, x⟩ : Σk, E k) ⟨j, y⟩ := Sigma.one_le_dist_of_ne hij _ _
        _ < 1 := by rw [h]; exact zero_lt_one


/-- The injection of a space in a disjoint union is an isometry -/
theorem isometry_mk (i : ι) : Isometry (Sigma.mk i : E i → Σk, E k) :=
                                    /-
                                      ι : Type u_1
                                      E : ι → Type u_2
                                      inst✝ : (i : ι) → MetricSpace (E i)
                                      i : ι
                                      x y : E i
                                      ⊢ Eq (Dist.dist ⟨i, x⟩ ⟨i, y⟩) (Dist.dist x y)
                                    -/
  Isometry.of_dist_eq fun x y => by simp
                                    /-
                                      🎉 no goals
                                    -/


/-- A disjoint union of complete metric spaces is complete. -/
protected theorem completeSpace [∀ i, CompleteSpace (E i)] : CompleteSpace (Σi, E i) := by
  /-
    ι : Type u_1
    E : ι → Type u_2
    inst✝¹ : (i : ι) → MetricSpace (E i)
    inst✝ : ∀ (i : ι), CompleteSpace (E i)
    ⊢ CompleteSpace (Sigma fun i => E i)
  -/
  set s : ι → Set (Σi, E i) := fun i => Sigma.fst ⁻¹' {i}
  /-
    ι : Type u_1
    E : ι → Type u_2
    inst✝¹ : (i : ι) → MetricSpace (E i)
    inst✝ : ∀ (i : ι), CompleteSpace (E i)
    s : ι → Set (Sigma fun i => E i) := fun i => Set.preimage Sigma.fst (Singleton …
    ⊢ CompleteSpace (Sigma fun i => E i)
  -/
  set U := { p : (Σk, E k) × Σk, E k | dist p.1 p.2 < 1 }
  have hc : ∀ i, IsComplete (s i) := fun i => by
    simp only [s, ← range_sigmaMk]
    exact (isometry_mk i).isUniformInducing.isComplete_range
  have hd : ∀ (i j), ∀ x ∈ s i, ∀ y ∈ s j, (x, y) ∈ U → i = j := fun i j x hx y hy hxy =>
    (Eq.symm hx).trans ((fst_eq_of_dist_lt_one _ _ hxy).trans hy)
  /-
    ι : Type u_1
    E : ι → Type u_2
    inst✝¹ : (i : ι) → MetricSpace (E i)
    inst✝ : ∀ (i : ι), CompleteSpace (E i)
    s : ι → Set (Sigma fun i => E i) := fun i => Set.preimage Sigma.fst (Singleton …
    U : Set (Prod (Sigma fun k => E k) (Sigma fun k => E k)) := setOf fun p => LT. …
    hc : ∀ (i : ι), IsComplete (s i)
    hd : ∀ (i j : ι) (x : Sigma fun i => E i), Membership.mem (s i) x → ∀ (y : Sig …
    ⊢ CompleteSpace (Sigma fun i => E i)
  -/
  refine completeSpace_of_isComplete_univ ?_
  /-
    ι : Type u_1
    E : ι → Type u_2
    inst✝¹ : (i : ι) → MetricSpace (E i)
    inst✝ : ∀ (i : ι), CompleteSpace (E i)
    s : ι → Set (Sigma fun i => E i) := fun i => Set.preimage Sigma.fst (Singleton …
    U : Set (Prod (Sigma fun k => E k) (Sigma fun k => E k)) := setOf fun p => LT. …
    hc : ∀ (i : ι), IsComplete (s i)
    hd : ∀ (i j : ι) (x : Sigma fun i => E i), Membership.mem (s i) x → ∀ (y : Sig …
    ⊢ IsComplete Set.univ
  -/
  convert isComplete_iUnion_separated hc (dist_mem_uniformity zero_lt_one) hd
  /-
    case h.e'_3
    ι : Type u_1
    E : ι → Type u_2
    inst✝¹ : (i : ι) → MetricSpace (E i)
    inst✝ : ∀ (i : ι), CompleteSpace (E i)
    s : ι → Set (Sigma fun i => E i) := fun i => Set.preimage Sigma.fst (Singleton …
    U : Set (Prod (Sigma fun k => E k) (Sigma fun k => E k)) := setOf fun p => LT. …
    hc : ∀ (i : ι), IsComplete (s i)
    hd : ∀ (i j : ι) (x : Sigma fun i => E i), Membership.mem (s i) x → ∀ (y : Sig …
    ⊢ Eq Set.univ (Set.iUnion fun i => s i)
  -/
  simp only [s, ← preimage_iUnion, iUnion_of_singleton, preimage_univ]
  /-
    🎉 no goals
  -/


/-- Given two isometric embeddings `Φ : Z → X` and `Ψ : Z → Y`, we define a pseudo metric space
structure on `X ⊕ Y` by declaring that `Φ x` and `Ψ x` are at distance `0`. -/
def gluePremetric (hΦ : Isometry Φ) (hΨ : Isometry Ψ) : PseudoMetricSpace (X ⊕ Y) where
  dist := glueDist Φ Ψ 0
  dist_self := glueDist_self Φ Ψ 0
  dist_comm := glueDist_comm Φ Ψ 0
                                                         /-
                                                           X : Type u
                                                           Y : Type v
                                                           Z : Type w
                                                           inst✝³ : Nonempty Z
                                                           inst✝² : MetricSpace Z
                                                           inst✝¹ : MetricSpace X
                                                           inst✝ : MetricSpace Y
                                                           Φ : Z → X
                                                           Ψ : Z → Y
                                                           ε : Real
                                                           hΦ : Isometry Φ
                                                           hΨ : Isometry Ψ
                                                           p q : Z
                                                           ⊢ LE.le (abs (HSub.hSub (Dist.dist (Φ p) (Φ q)) (Dist.dist (Ψ p) (Ψ q)))) (HMu …
                                                         -/
  dist_triangle := glueDist_triangle Φ Ψ 0 fun p q => by rw [hΦ.dist_eq, hΨ.dist_eq]; simp
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


/-- Given two isometric embeddings `Φ : Z → X` and `Ψ : Z → Y`, we define a
space `GlueSpace hΦ hΨ` by identifying in `X ⊕ Y` the points `Φ x` and `Ψ x`. -/
def GlueSpace (hΦ : Isometry Φ) (hΨ : Isometry Ψ) : Type _ :=
  @SeparationQuotient _ (gluePremetric hΦ hΨ).toUniformSpace.toTopologicalSpace


instance (hΦ : Isometry Φ) (hΨ : Isometry Ψ) : MetricSpace (GlueSpace hΦ hΨ) :=
  inferInstanceAs <| MetricSpace <|
    @SeparationQuotient _ (gluePremetric hΦ hΨ).toUniformSpace.toTopologicalSpace


/-- The canonical map from `X` to the space obtained by gluing isometric subsets in `X` and `Y`. -/
def toGlueL (hΦ : Isometry Φ) (hΨ : Isometry Ψ) (x : X) : GlueSpace hΦ hΨ :=
  Quotient.mk'' (.inl x)


/-- The canonical map from `Y` to the space obtained by gluing isometric subsets in `X` and `Y`. -/
def toGlueR (hΦ : Isometry Φ) (hΨ : Isometry Ψ) (y : Y) : GlueSpace hΦ hΨ :=
  Quotient.mk'' (.inr y)


instance inhabitedLeft (hΦ : Isometry Φ) (hΨ : Isometry Ψ) [Inhabited X] :
    Inhabited (GlueSpace hΦ hΨ) :=
  ⟨toGlueL _ _ default⟩


instance inhabitedRight (hΦ : Isometry Φ) (hΨ : Isometry Ψ) [Inhabited Y] :
    Inhabited (GlueSpace hΦ hΨ) :=
  ⟨toGlueR _ _ default⟩


theorem toGlue_commute (hΦ : Isometry Φ) (hΨ : Isometry Ψ) :
    toGlueL hΦ hΨ ∘ Φ = toGlueR hΦ hΨ ∘ Ψ := by
  /-
    X : Type u
    Y : Type v
    Z : Type w
    inst✝³ : Nonempty Z
    inst✝² : MetricSpace Z
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    Φ : Z → X
    Ψ : Z → Y
    hΦ : Isometry Φ
    hΨ : Isometry Ψ
    ⊢ Eq (Function.comp (Metric.toGlueL hΦ hΨ) Φ) (Function.comp (Metric.toGlueR h …
  -/
  let i : PseudoMetricSpace (X ⊕ Y) := gluePremetric hΦ hΨ
  /-
    X : Type u
    Y : Type v
    Z : Type w
    inst✝³ : Nonempty Z
    inst✝² : MetricSpace Z
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    Φ : Z → X
    Ψ : Z → Y
    hΦ : Isometry Φ
    hΨ : Isometry Ψ
    i : PseudoMetricSpace (Sum X Y) := Metric.gluePremetric hΦ hΨ
    ⊢ Eq (Function.comp (Metric.toGlueL hΦ hΨ) Φ) (Function.comp (Metric.toGlueR h …
  -/
  let _ := i.toUniformSpace.toTopologicalSpace
  /-
    X : Type u
    Y : Type v
    Z : Type w
    inst✝³ : Nonempty Z
    inst✝² : MetricSpace Z
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    Φ : Z → X
    Ψ : Z → Y
    hΦ : Isometry Φ
    hΨ : Isometry Ψ
    i : PseudoMetricSpace (Sum X Y) := Metric.gluePremetric hΦ hΨ
    x✝ : TopologicalSpace (Sum X Y) := UniformSpace.toTopologicalSpace
    ⊢ Eq (Function.comp (Metric.toGlueL hΦ hΨ) Φ) (Function.comp (Metric.toGlueR h …
  -/
  funext
  /-
    case h
    X : Type u
    Y : Type v
    Z : Type w
    inst✝³ : Nonempty Z
    inst✝² : MetricSpace Z
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    Φ : Z → X
    Ψ : Z → Y
    hΦ : Isometry Φ
    hΨ : Isometry Ψ
    i : PseudoMetricSpace (Sum X Y) := Metric.gluePremetric hΦ hΨ
    x✝¹ : TopologicalSpace (Sum X Y) := UniformSpace.toTopologicalSpace
    x✝ : Z
    ⊢ Eq (Function.comp (Metric.toGlueL hΦ hΨ) Φ x✝) (Function.comp (Metric.toGlue …
  -/
  simp only [comp, toGlueL, toGlueR]
  /-
    case h
    X : Type u
    Y : Type v
    Z : Type w
    inst✝³ : Nonempty Z
    inst✝² : MetricSpace Z
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    Φ : Z → X
    Ψ : Z → Y
    hΦ : Isometry Φ
    hΨ : Isometry Ψ
    i : PseudoMetricSpace (Sum X Y) := Metric.gluePremetric hΦ hΨ
    x✝¹ : TopologicalSpace (Sum X Y) := UniformSpace.toTopologicalSpace
    x✝ : Z
    ⊢ Eq (Quotient.mk'' (Sum.inl (Φ x✝))) (Quotient.mk'' (Sum.inr (Ψ x✝)))
  -/
  refine SeparationQuotient.mk_eq_mk.2 (Metric.inseparable_iff.2 ?_)
  /-
    case h
    X : Type u
    Y : Type v
    Z : Type w
    inst✝³ : Nonempty Z
    inst✝² : MetricSpace Z
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    Φ : Z → X
    Ψ : Z → Y
    hΦ : Isometry Φ
    hΨ : Isometry Ψ
    i : PseudoMetricSpace (Sum X Y) := Metric.gluePremetric hΦ hΨ
    x✝¹ : TopologicalSpace (Sum X Y) := UniformSpace.toTopologicalSpace
    x✝ : Z
    ⊢ Eq (Dist.dist (Sum.inl (Φ x✝)) (Sum.inr (Ψ x✝))) 0
  -/
  exact glueDist_glued_points Φ Ψ 0 _
  /-
    🎉 no goals
  -/


theorem toGlueL_isometry (hΦ : Isometry Φ) (hΨ : Isometry Ψ) : Isometry (toGlueL hΦ hΨ) :=
  Isometry.of_dist_eq fun _ _ => rfl


theorem toGlueR_isometry (hΦ : Isometry Φ) (hΨ : Isometry Ψ) : Isometry (toGlueR hΦ hΨ) :=
  Isometry.of_dist_eq fun _ _ => rfl


/-- Predistance on the disjoint union `Σ n, X n`. -/
def inductiveLimitDist (f : ∀ n, X n → X (n + 1)) (x y : Σn, X n) : ℝ :=
  dist (leRecOn (le_max_left x.1 y.1) (f _) x.2 : X (max x.1 y.1))
    (leRecOn (le_max_right x.1 y.1) (f _) y.2 : X (max x.1 y.1))


/-- The predistance on the disjoint union `Σ n, X n` can be computed in any `X k` for large
enough `k`. -/
theorem inductiveLimitDist_eq_dist (I : ∀ n, Isometry (f n)) (x y : Σn, X n) :
    ∀ m (hx : x.1 ≤ m) (hy : y.1 ≤ m), inductiveLimitDist f x y =
      dist (leRecOn hx (f _) x.2 : X m) (leRecOn hy (f _) y.2 : X m)
  | 0, hx, hy => by
    /-
      X : Nat → Type u
      inst✝ : (n : Nat) → MetricSpace (X n)
      f : (n : Nat) → X n → X (HAdd.hAdd n 1)
      I : ∀ (n : Nat), Isometry (f n)
      x y : Sigma fun n => X n
      hx : LE.le x.fst 0
      hy : LE.le y.fst 0
      ⊢ Eq (Metric.inductiveLimitDist f x y) (Dist.dist (Nat.leRecOn hx (fun {k} =>  …
    -/
    cases' x with i x; cases' y with j y
    /-
      case mk.mk
      X : Nat → Type u
      inst✝ : (n : Nat) → MetricSpace (X n)
      f : (n : Nat) → X n → X (HAdd.hAdd n 1)
      I : ∀ (n : Nat), Isometry (f n)
      i : Nat
      x : X i
      hx : LE.le ⟨i, x⟩.fst 0
      j : Nat
      y : X j
      hy : LE.le ⟨j, y⟩.fst 0
      ⊢ Eq (Metric.inductiveLimitDist f ⟨i, x⟩ ⟨j, y⟩) (Dist.dist (Nat.leRecOn hx (f …
    -/
    obtain rfl : i = 0 := nonpos_iff_eq_zero.1 hx
    /-
      case mk.mk
      X : Nat → Type u
      inst✝ : (n : Nat) → MetricSpace (X n)
      f : (n : Nat) → X n → X (HAdd.hAdd n 1)
      I : ∀ (n : Nat), Isometry (f n)
      j : Nat
      y : X j
      hy : LE.le ⟨j, y⟩.fst 0
      x : X 0
      hx : LE.le ⟨0, x⟩.fst 0
      ⊢ Eq (Metric.inductiveLimitDist f ⟨0, x⟩ ⟨j, y⟩) (Dist.dist (Nat.leRecOn hx (f …
    -/
    obtain rfl : j = 0 := nonpos_iff_eq_zero.1 hy
    /-
      case mk.mk
      X : Nat → Type u
      inst✝ : (n : Nat) → MetricSpace (X n)
      f : (n : Nat) → X n → X (HAdd.hAdd n 1)
      I : ∀ (n : Nat), Isometry (f n)
      x : X 0
      hx : LE.le ⟨0, x⟩.fst 0
      y : X 0
      hy : LE.le ⟨0, y⟩.fst 0
      ⊢ Eq (Metric.inductiveLimitDist f ⟨0, x⟩ ⟨0, y⟩) (Dist.dist (Nat.leRecOn hx (f …
    -/
    rfl
    /-
      🎉 no goals
    -/
  | (m + 1), hx, hy => by
    /-
      X : Nat → Type u
      inst✝ : (n : Nat) → MetricSpace (X n)
      f : (n : Nat) → X n → X (HAdd.hAdd n 1)
      I : ∀ (n : Nat), Isometry (f n)
      x y : Sigma fun n => X n
      m : Nat
      hx : LE.le x.fst (HAdd.hAdd m 1)
      hy : LE.le y.fst (HAdd.hAdd m 1)
      ⊢ Eq (Metric.inductiveLimitDist f x y) (Dist.dist (Nat.leRecOn hx (fun {k} =>  …
    -/
    by_cases h : max x.1 y.1 = (m + 1)
      /-
        case pos
        X : Nat → Type u
        inst✝ : (n : Nat) → MetricSpace (X n)
        f : (n : Nat) → X n → X (HAdd.hAdd n 1)
        I : ∀ (n : Nat), Isometry (f n)
        x y : Sigma fun n => X n
        m : Nat
        hx : LE.le x.fst (HAdd.hAdd m 1)
        hy : LE.le y.fst (HAdd.hAdd m 1)
        h : Eq (Max.max x.fst y.fst) (HAdd.hAdd m 1)
        ⊢ Eq (Metric.inductiveLimitDist f x y) (Dist.dist (Nat.leRecOn hx (fun {k} =>  …
      -/
    · generalize m + 1 = m' at *
      /-
        case pos
        X : Nat → Type u
        inst✝ : (n : Nat) → MetricSpace (X n)
        f : (n : Nat) → X n → X (HAdd.hAdd n 1)
        I : ∀ (n : Nat), Isometry (f n)
        x y : Sigma fun n => X n
        m m' : Nat
        hx : LE.le x.fst m'
        hy : LE.le y.fst m'
        h : Eq (Max.max x.fst y.fst) m'
        ⊢ Eq (Metric.inductiveLimitDist f x y) (Dist.dist (Nat.leRecOn hx (fun {k} =>  …
      -/
      subst m'
      /-
        case pos
        X : Nat → Type u
        inst✝ : (n : Nat) → MetricSpace (X n)
        f : (n : Nat) → X n → X (HAdd.hAdd n 1)
        I : ∀ (n : Nat), Isometry (f n)
        x y : Sigma fun n => X n
        m : Nat
        hx : LE.le x.fst (Max.max x.fst y.fst)
        hy : LE.le y.fst (Max.max x.fst y.fst)
        ⊢ Eq (Metric.inductiveLimitDist f x y) (Dist.dist (Nat.leRecOn hx (fun {k} =>  …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case neg
        X : Nat → Type u
        inst✝ : (n : Nat) → MetricSpace (X n)
        f : (n : Nat) → X n → X (HAdd.hAdd n 1)
        I : ∀ (n : Nat), Isometry (f n)
        x y : Sigma fun n => X n
        m : Nat
        hx : LE.le x.fst (HAdd.hAdd m 1)
        hy : LE.le y.fst (HAdd.hAdd m 1)
        h : Not (Eq (Max.max x.fst y.fst) (HAdd.hAdd m 1))
        ⊢ Eq (Metric.inductiveLimitDist f x y) (Dist.dist (Nat.leRecOn hx (fun {k} =>  …
      -/
    · have : max x.1 y.1 ≤ succ m := by simp [hx, hy]
      /-
        case neg
        X : Nat → Type u
        inst✝ : (n : Nat) → MetricSpace (X n)
        f : (n : Nat) → X n → X (HAdd.hAdd n 1)
        I : ∀ (n : Nat), Isometry (f n)
        x y : Sigma fun n => X n
        m : Nat
        hx : LE.le x.fst (HAdd.hAdd m 1)
        hy : LE.le y.fst (HAdd.hAdd m 1)
        h : Not (Eq (Max.max x.fst y.fst) (HAdd.hAdd m 1))
        this : LE.le (Max.max x.fst y.fst) m.succ
        ⊢ Eq (Metric.inductiveLimitDist f x y) (Dist.dist (Nat.leRecOn hx (fun {k} =>  …
      -/
      have : max x.1 y.1 ≤ m := by simpa [h] using of_le_succ this
      /-
        case neg
        X : Nat → Type u
        inst✝ : (n : Nat) → MetricSpace (X n)
        f : (n : Nat) → X n → X (HAdd.hAdd n 1)
        I : ∀ (n : Nat), Isometry (f n)
        x y : Sigma fun n => X n
        m : Nat
        hx : LE.le x.fst (HAdd.hAdd m 1)
        hy : LE.le y.fst (HAdd.hAdd m 1)
        h : Not (Eq (Max.max x.fst y.fst) (HAdd.hAdd m 1))
        this✝ : LE.le (Max.max x.fst y.fst) m.succ
        this : LE.le (Max.max x.fst y.fst) m
        ⊢ Eq (Metric.inductiveLimitDist f x y) (Dist.dist (Nat.leRecOn hx (fun {k} =>  …
      -/
      have xm : x.1 ≤ m := le_trans (le_max_left _ _) this
      /-
        case neg
        X : Nat → Type u
        inst✝ : (n : Nat) → MetricSpace (X n)
        f : (n : Nat) → X n → X (HAdd.hAdd n 1)
        I : ∀ (n : Nat), Isometry (f n)
        x y : Sigma fun n => X n
        m : Nat
        hx : LE.le x.fst (HAdd.hAdd m 1)
        hy : LE.le y.fst (HAdd.hAdd m 1)
        h : Not (Eq (Max.max x.fst y.fst) (HAdd.hAdd m 1))
        this✝ : LE.le (Max.max x.fst y.fst) m.succ
        this : LE.le (Max.max x.fst y.fst) m
        xm : LE.le x.fst m
        ⊢ Eq (Metric.inductiveLimitDist f x y) (Dist.dist (Nat.leRecOn hx (fun {k} =>  …
      -/
      have ym : y.1 ≤ m := le_trans (le_max_right _ _) this
      /-
        case neg
        X : Nat → Type u
        inst✝ : (n : Nat) → MetricSpace (X n)
        f : (n : Nat) → X n → X (HAdd.hAdd n 1)
        I : ∀ (n : Nat), Isometry (f n)
        x y : Sigma fun n => X n
        m : Nat
        hx : LE.le x.fst (HAdd.hAdd m 1)
        hy : LE.le y.fst (HAdd.hAdd m 1)
        h : Not (Eq (Max.max x.fst y.fst) (HAdd.hAdd m 1))
        this✝ : LE.le (Max.max x.fst y.fst) m.succ
        this : LE.le (Max.max x.fst y.fst) m
        xm : LE.le x.fst m
        ym : LE.le y.fst m
        ⊢ Eq (Metric.inductiveLimitDist f x y) (Dist.dist (Nat.leRecOn hx (fun {k} =>  …
      -/
      rw [leRecOn_succ xm, leRecOn_succ ym, (I m).dist_eq]
      /-
        case neg
        X : Nat → Type u
        inst✝ : (n : Nat) → MetricSpace (X n)
        f : (n : Nat) → X n → X (HAdd.hAdd n 1)
        I : ∀ (n : Nat), Isometry (f n)
        x y : Sigma fun n => X n
        m : Nat
        hx : LE.le x.fst (HAdd.hAdd m 1)
        hy : LE.le y.fst (HAdd.hAdd m 1)
        h : Not (Eq (Max.max x.fst y.fst) (HAdd.hAdd m 1))
        this✝ : LE.le (Max.max x.fst y.fst) m.succ
        this : LE.le (Max.max x.fst y.fst) m
        xm : LE.le x.fst m
        ym : LE.le y.fst m
        ⊢ Eq (Metric.inductiveLimitDist f x y) (Dist.dist (Nat.leRecOn xm (fun {k} =>  …
      -/
      exact inductiveLimitDist_eq_dist I x y m xm ym
      /-
        🎉 no goals
      -/


/-- Premetric space structure on `Σ n, X n`. -/
def inductivePremetric (I : ∀ n, Isometry (f n)) : PseudoMetricSpace (Σn, X n) where
  dist := inductiveLimitDist f
                    /-
                      X : Nat → Type u
                      inst✝ : (n : Nat) → MetricSpace (X n)
                      f : (n : Nat) → X n → X (HAdd.hAdd n 1)
                      I : ∀ (n : Nat), Isometry (f n)
                      x : Sigma fun n => X n
                      ⊢ Eq (Dist.dist x x) 0
                    -/
  dist_self x := by simp [dist, inductiveLimitDist]
                    /-
                      🎉 no goals
                    -/
  dist_comm x y := by
    /-
      X : Nat → Type u
      inst✝ : (n : Nat) → MetricSpace (X n)
      f : (n : Nat) → X n → X (HAdd.hAdd n 1)
      I : ∀ (n : Nat), Isometry (f n)
      x y : Sigma fun n => X n
      ⊢ Eq (Dist.dist x y) (Dist.dist y x)
    -/
    let m := max x.1 y.1
    /-
      X : Nat → Type u
      inst✝ : (n : Nat) → MetricSpace (X n)
      f : (n : Nat) → X n → X (HAdd.hAdd n 1)
      I : ∀ (n : Nat), Isometry (f n)
      x y : Sigma fun n => X n
      m : Nat := Max.max x.fst y.fst
      ⊢ Eq (Dist.dist x y) (Dist.dist y x)
    -/
    have hx : x.1 ≤ m := le_max_left _ _
    /-
      X : Nat → Type u
      inst✝ : (n : Nat) → MetricSpace (X n)
      f : (n : Nat) → X n → X (HAdd.hAdd n 1)
      I : ∀ (n : Nat), Isometry (f n)
      x y : Sigma fun n => X n
      m : Nat := Max.max x.fst y.fst
      hx : LE.le x.fst m
      ⊢ Eq (Dist.dist x y) (Dist.dist y x)
    -/
    have hy : y.1 ≤ m := le_max_right _ _
    /-
      X : Nat → Type u
      inst✝ : (n : Nat) → MetricSpace (X n)
      f : (n : Nat) → X n → X (HAdd.hAdd n 1)
      I : ∀ (n : Nat), Isometry (f n)
      x y : Sigma fun n => X n
      m : Nat := Max.max x.fst y.fst
      hx : LE.le x.fst m
      hy : LE.le y.fst m
      ⊢ Eq (Dist.dist x y) (Dist.dist y x)
    -/
    unfold dist; simp only
    rw [inductiveLimitDist_eq_dist I x y m hx hy, inductiveLimitDist_eq_dist I y x m hy hx,
      dist_comm]
  dist_triangle x y z := by
    /-
      X : Nat → Type u
      inst✝ : (n : Nat) → MetricSpace (X n)
      f : (n : Nat) → X n → X (HAdd.hAdd n 1)
      I : ∀ (n : Nat), Isometry (f n)
      x y z : Sigma fun n => X n
      ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
    -/
    let m := max (max x.1 y.1) z.1
    /-
      X : Nat → Type u
      inst✝ : (n : Nat) → MetricSpace (X n)
      f : (n : Nat) → X n → X (HAdd.hAdd n 1)
      I : ∀ (n : Nat), Isometry (f n)
      x y z : Sigma fun n => X n
      m : Nat := Max.max (Max.max x.fst y.fst) z.fst
      ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
    -/
    have hx : x.1 ≤ m := le_trans (le_max_left _ _) (le_max_left _ _)
    /-
      X : Nat → Type u
      inst✝ : (n : Nat) → MetricSpace (X n)
      f : (n : Nat) → X n → X (HAdd.hAdd n 1)
      I : ∀ (n : Nat), Isometry (f n)
      x y z : Sigma fun n => X n
      m : Nat := Max.max (Max.max x.fst y.fst) z.fst
      hx : LE.le x.fst m
      ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
    -/
    have hy : y.1 ≤ m := le_trans (le_max_right _ _) (le_max_left _ _)
    /-
      X : Nat → Type u
      inst✝ : (n : Nat) → MetricSpace (X n)
      f : (n : Nat) → X n → X (HAdd.hAdd n 1)
      I : ∀ (n : Nat), Isometry (f n)
      x y z : Sigma fun n => X n
      m : Nat := Max.max (Max.max x.fst y.fst) z.fst
      hx : LE.le x.fst m
      hy : LE.le y.fst m
      ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
    -/
    have hz : z.1 ≤ m := le_max_right _ _
    calc
      inductiveLimitDist f x z = dist (leRecOn hx (f _) x.2 : X m) (leRecOn hz (f _) z.2 : X m) :=
        inductiveLimitDist_eq_dist I x z m hx hz
      _ ≤ dist (leRecOn hx (f _) x.2 : X m) (leRecOn hy (f _) y.2 : X m) +
            dist (leRecOn hy (f _) y.2 : X m) (leRecOn hz (f _) z.2 : X m) :=
        (dist_triangle _ _ _)
      _ = inductiveLimitDist f x y + inductiveLimitDist f y z := by
        rw [inductiveLimitDist_eq_dist I x y m hx hy, inductiveLimitDist_eq_dist I y z m hy hz]


/-- The type giving the inductive limit in a metric space context. -/
def InductiveLimit (I : ∀ n, Isometry (f n)) : Type _ :=
  @SeparationQuotient _ (inductivePremetric I).toUniformSpace.toTopologicalSpace


instance {I : ∀ (n : ℕ), Isometry (f n)} : MetricSpace (InductiveLimit (f := f) I) :=
  inferInstanceAs <| MetricSpace <|
    @SeparationQuotient _ (inductivePremetric I).toUniformSpace.toTopologicalSpace


/-- Mapping each `X n` to the inductive limit. -/
def toInductiveLimit (I : ∀ n, Isometry (f n)) (n : ℕ) (x : X n) : Metric.InductiveLimit I :=
  Quotient.mk'' (Sigma.mk n x)


instance (I : ∀ n, Isometry (f n)) [Inhabited (X 0)] : Inhabited (InductiveLimit I) :=
  ⟨toInductiveLimit _ 0 default⟩


/-- The map `toInductiveLimit n` mapping `X n` to the inductive limit is an isometry. -/
theorem toInductiveLimit_isometry (I : ∀ n, Isometry (f n)) (n : ℕ) :
    Isometry (toInductiveLimit I n) :=
  Isometry.of_dist_eq fun x y => by
    /-
      X : Nat → Type u
      inst✝ : (n : Nat) → MetricSpace (X n)
      f : (n : Nat) → X n → X (HAdd.hAdd n 1)
      I : ∀ (n : Nat), Isometry (f n)
      n : Nat
      x y : X n
      ⊢ Eq (Dist.dist (Metric.toInductiveLimit I n x) (Metric.toInductiveLimit I n y …
    -/
    change inductiveLimitDist f ⟨n, x⟩ ⟨n, y⟩ = dist x y
    rw [inductiveLimitDist_eq_dist I ⟨n, x⟩ ⟨n, y⟩ n (le_refl n) (le_refl n), leRecOn_self,
      leRecOn_self]


/-- The maps `toInductiveLimit n` are compatible with the maps `f n`. -/
theorem toInductiveLimit_commute (I : ∀ n, Isometry (f n)) (n : ℕ) :
    toInductiveLimit I n.succ ∘ f n = toInductiveLimit I n := by
  /-
    X : Nat → Type u
    inst✝ : (n : Nat) → MetricSpace (X n)
    f : (n : Nat) → X n → X (HAdd.hAdd n 1)
    I : ∀ (n : Nat), Isometry (f n)
    n : Nat
    ⊢ Eq (Function.comp (Metric.toInductiveLimit I n.succ) (f n)) (Metric.toInduct …
  -/
  let h := inductivePremetric I
  /-
    X : Nat → Type u
    inst✝ : (n : Nat) → MetricSpace (X n)
    f : (n : Nat) → X n → X (HAdd.hAdd n 1)
    I : ∀ (n : Nat), Isometry (f n)
    n : Nat
    h : PseudoMetricSpace (Sigma fun n => X n) := Metric.inductivePremetric I
    ⊢ Eq (Function.comp (Metric.toInductiveLimit I n.succ) (f n)) (Metric.toInduct …
  -/
  let _ := h.toUniformSpace.toTopologicalSpace
  /-
    X : Nat → Type u
    inst✝ : (n : Nat) → MetricSpace (X n)
    f : (n : Nat) → X n → X (HAdd.hAdd n 1)
    I : ∀ (n : Nat), Isometry (f n)
    n : Nat
    h : PseudoMetricSpace (Sigma fun n => X n) := Metric.inductivePremetric I
    x✝ : TopologicalSpace (Sigma fun n => X n) := UniformSpace.toTopologicalSpace
    ⊢ Eq (Function.comp (Metric.toInductiveLimit I n.succ) (f n)) (Metric.toInduct …
  -/
  funext x
  /-
    case h
    X : Nat → Type u
    inst✝ : (n : Nat) → MetricSpace (X n)
    f : (n : Nat) → X n → X (HAdd.hAdd n 1)
    I : ∀ (n : Nat), Isometry (f n)
    n : Nat
    h : PseudoMetricSpace (Sigma fun n => X n) := Metric.inductivePremetric I
    x✝ : TopologicalSpace (Sigma fun n => X n) := UniformSpace.toTopologicalSpace
    x : X n
    ⊢ Eq (Function.comp (Metric.toInductiveLimit I n.succ) (f n) x) (Metric.toIndu …
  -/
  simp only [comp, toInductiveLimit]
  /-
    case h
    X : Nat → Type u
    inst✝ : (n : Nat) → MetricSpace (X n)
    f : (n : Nat) → X n → X (HAdd.hAdd n 1)
    I : ∀ (n : Nat), Isometry (f n)
    n : Nat
    h : PseudoMetricSpace (Sigma fun n => X n) := Metric.inductivePremetric I
    x✝ : TopologicalSpace (Sigma fun n => X n) := UniformSpace.toTopologicalSpace
    x : X n
    ⊢ Eq (Quotient.mk'' ⟨n.succ, f n x⟩) (Quotient.mk'' ⟨n, x⟩)
  -/
  refine SeparationQuotient.mk_eq_mk.2 (Metric.inseparable_iff.2 ?_)
  /-
    case h
    X : Nat → Type u
    inst✝ : (n : Nat) → MetricSpace (X n)
    f : (n : Nat) → X n → X (HAdd.hAdd n 1)
    I : ∀ (n : Nat), Isometry (f n)
    n : Nat
    h : PseudoMetricSpace (Sigma fun n => X n) := Metric.inductivePremetric I
    x✝ : TopologicalSpace (Sigma fun n => X n) := UniformSpace.toTopologicalSpace
    x : X n
    ⊢ Eq (Dist.dist ⟨n.succ, f n x⟩ ⟨n, x⟩) 0
  -/
  show inductiveLimitDist f ⟨n.succ, f n x⟩ ⟨n, x⟩ = 0
  rw [inductiveLimitDist_eq_dist I ⟨n.succ, f n x⟩ ⟨n, x⟩ n.succ, leRecOn_self,
    leRecOn_succ, leRecOn_self, dist_self]
  /-
    case h.h2
    X : Nat → Type u
    inst✝ : (n : Nat) → MetricSpace (X n)
    f : (n : Nat) → X n → X (HAdd.hAdd n 1)
    I : ∀ (n : Nat), Isometry (f n)
    n : Nat
    h : PseudoMetricSpace (Sigma fun n => X n) := Metric.inductivePremetric I
    x✝ : TopologicalSpace (Sigma fun n => X n) := UniformSpace.toTopologicalSpace
    x : X n
    ⊢ LE.le ⟨n, x⟩.fst (HAdd.hAdd n 1)
  -/
  exact le_succ _
  /-
    🎉 no goals
  -/


