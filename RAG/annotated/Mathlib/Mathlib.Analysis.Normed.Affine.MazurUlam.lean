/-- If an isometric self-homeomorphism of a normed vector space over `ℝ` fixes `x` and `y`,
then it fixes the midpoint of `[x, y]`. This is a lemma for a more general Mazur-Ulam theorem,
see below. -/
theorem midpoint_fixed {x y : PE} :
    ∀ e : PE ≃ᵢ PE, e x = x → e y = y → e (midpoint ℝ x y) = midpoint ℝ x y := by
  /-
    E : Type u_1
    PE : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : MetricSpace PE
    inst✝ : NormedAddTorsor E PE
    x y : PE
    ⊢ ∀ (e : IsometryEquiv PE PE), Eq (e x) x → Eq (e y) y → Eq (e (midpoint Real  …
  -/
  set z := midpoint ℝ x y
  -- Consider the set of `e : E ≃ᵢ E` such that `e x = x` and `e y = y`
  /-
    E : Type u_1
    PE : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : MetricSpace PE
    inst✝ : NormedAddTorsor E PE
    x y : PE
    z : PE := midpoint Real x y
    ⊢ ∀ (e : IsometryEquiv PE PE), Eq (e x) x → Eq (e y) y → Eq (e z) z
  -/
  set s := { e : PE ≃ᵢ PE | e x = x ∧ e y = y }
  /-
    E : Type u_1
    PE : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : MetricSpace PE
    inst✝ : NormedAddTorsor E PE
    x y : PE
    z : PE := midpoint Real x y
    s : Set (IsometryEquiv PE PE) := setOf fun e => And (Eq (e x) x) (Eq (e y) y)
    ⊢ ∀ (e : IsometryEquiv PE PE), Eq (e x) x → Eq (e y) y → Eq (e z) z
  -/
  haveI : Nonempty s := ⟨⟨IsometryEquiv.refl PE, rfl, rfl⟩⟩
  -- On the one hand, `e` cannot send the midpoint `z` of `[x, y]` too far
  have h_bdd : BddAbove (range fun e : s => dist ((e : PE ≃ᵢ PE) z) z) := by
    refine ⟨dist x z + dist x z, forall_mem_range.2 <| Subtype.forall.2 ?_⟩
    rintro e ⟨hx, _⟩
    calc
      dist (e z) z ≤ dist (e z) x + dist x z := dist_triangle (e z) x z
      _ = dist (e x) (e z) + dist x z := by rw [hx, dist_comm]
      _ = dist x z + dist x z := by rw [e.dist_eq x z]
  -- On the other hand, consider the map `f : (E ≃ᵢ E) → (E ≃ᵢ E)`
  -- sending each `e` to `R ∘ e⁻¹ ∘ R ∘ e`, where `R` is the point reflection in the
  -- midpoint `z` of `[x, y]`.
  /-
    E : Type u_1
    PE : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : MetricSpace PE
    inst✝ : NormedAddTorsor E PE
    x y : PE
    z : PE := midpoint Real x y
    s : Set (IsometryEquiv PE PE) := setOf fun e => And (Eq (e x) x) (Eq (e y) y)
    this : Nonempty ↑s
    h_bdd : BddAbove (Set.range fun e => Dist.dist (↑e z) z)
    ⊢ ∀ (e : IsometryEquiv PE PE), Eq (e x) x → Eq (e y) y → Eq (e z) z
  -/
  set R : PE ≃ᵢ PE := (pointReflection ℝ z).toIsometryEquiv
  /-
    E : Type u_1
    PE : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : MetricSpace PE
    inst✝ : NormedAddTorsor E PE
    x y : PE
    z : PE := midpoint Real x y
    s : Set (IsometryEquiv PE PE) := setOf fun e => And (Eq (e x) x) (Eq (e y) y)
    this : Nonempty ↑s
    h_bdd : BddAbove (Set.range fun e => Dist.dist (↑e z) z)
    R : IsometryEquiv PE PE := (AffineIsometryEquiv.pointReflection Real z).toIsom …
    ⊢ ∀ (e : IsometryEquiv PE PE), Eq (e x) x → Eq (e y) y → Eq (e z) z
  -/
  set f : PE ≃ᵢ PE → PE ≃ᵢ PE := fun e => ((e.trans R).trans e.symm).trans R
  -- Note that `f` doubles the value of `dist (e z) z`
  have hf_dist : ∀ e, dist (f e z) z = 2 * dist (e z) z := by
    intro e
    dsimp only [trans_apply, coe_toIsometryEquiv, f, R]
    rw [dist_pointReflection_fixed, ← e.dist_eq, e.apply_symm_apply,
      dist_pointReflection_self_real, dist_comm]
  -- Also note that `f` maps `s` to itself
  have hf_maps_to : MapsTo f s s := by
    rintro e ⟨hx, hy⟩
    constructor <;> simp [f, R, z, hx, hy, e.symm_apply_eq.2 hx.symm, e.symm_apply_eq.2 hy.symm]
  -- Therefore, `dist (e z) z = 0` for all `e ∈ s`.
  /-
    E : Type u_1
    PE : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : MetricSpace PE
    inst✝ : NormedAddTorsor E PE
    x y : PE
    z : PE := midpoint Real x y
    s : Set (IsometryEquiv PE PE) := setOf fun e => And (Eq (e x) x) (Eq (e y) y)
    this : Nonempty ↑s
    h_bdd : BddAbove (Set.range fun e => Dist.dist (↑e z) z)
    R : IsometryEquiv PE PE := (AffineIsometryEquiv.pointReflection Real z).toIsom …
    f : IsometryEquiv PE PE → IsometryEquiv PE PE := fun e => ((e.trans R).trans e …
    hf_dist : ∀ (e : IsometryEquiv PE PE), Eq (Dist.dist ((f e) z) z) (HMul.hMul 2 …
    hf_maps_to : Set.MapsTo f s s
    ⊢ ∀ (e : IsometryEquiv PE PE), Eq (e x) x → Eq (e y) y → Eq (e z) z
  -/
  set c := ⨆ e : s, dist ((e : PE ≃ᵢ PE) z) z
  have : c ≤ c / 2 := by
    apply ciSup_le
    rintro ⟨e, he⟩
    simp only [Subtype.coe_mk, le_div_iff₀' (zero_lt_two' ℝ), ← hf_dist]
    exact le_ciSup h_bdd ⟨f e, hf_maps_to he⟩
  /-
    E : Type u_1
    PE : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : MetricSpace PE
    inst✝ : NormedAddTorsor E PE
    x y : PE
    z : PE := midpoint Real x y
    s : Set (IsometryEquiv PE PE) := setOf fun e => And (Eq (e x) x) (Eq (e y) y)
    this✝ : Nonempty ↑s
    h_bdd : BddAbove (Set.range fun e => Dist.dist (↑e z) z)
    R : IsometryEquiv PE PE := (AffineIsometryEquiv.pointReflection Real z).toIsom …
    f : IsometryEquiv PE PE → IsometryEquiv PE PE := fun e => ((e.trans R).trans e …
    hf_dist : ∀ (e : IsometryEquiv PE PE), Eq (Dist.dist ((f e) z) z) (HMul.hMul 2 …
    hf_maps_to : Set.MapsTo f s s
    c : Real := iSup fun e => Dist.dist (↑e z) z
    this : LE.le c (HDiv.hDiv c 2)
    ⊢ ∀ (e : IsometryEquiv PE PE), Eq (e x) x → Eq (e y) y → Eq (e z) z
  -/
  replace : c ≤ 0 := by linarith
  /-
    E : Type u_1
    PE : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : MetricSpace PE
    inst✝ : NormedAddTorsor E PE
    x y : PE
    z : PE := midpoint Real x y
    s : Set (IsometryEquiv PE PE) := setOf fun e => And (Eq (e x) x) (Eq (e y) y)
    this✝ : Nonempty ↑s
    h_bdd : BddAbove (Set.range fun e => Dist.dist (↑e z) z)
    R : IsometryEquiv PE PE := (AffineIsometryEquiv.pointReflection Real z).toIsom …
    f : IsometryEquiv PE PE → IsometryEquiv PE PE := fun e => ((e.trans R).trans e …
    hf_dist : ∀ (e : IsometryEquiv PE PE), Eq (Dist.dist ((f e) z) z) (HMul.hMul 2 …
    hf_maps_to : Set.MapsTo f s s
    c : Real := iSup fun e => Dist.dist (↑e z) z
    this : LE.le c 0
    ⊢ ∀ (e : IsometryEquiv PE PE), Eq (e x) x → Eq (e y) y → Eq (e z) z
  -/
  refine fun e hx hy => dist_le_zero.1 (le_trans ?_ this)
  /-
    E : Type u_1
    PE : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : MetricSpace PE
    inst✝ : NormedAddTorsor E PE
    x y : PE
    z : PE := midpoint Real x y
    s : Set (IsometryEquiv PE PE) := setOf fun e => And (Eq (e x) x) (Eq (e y) y)
    this✝ : Nonempty ↑s
    h_bdd : BddAbove (Set.range fun e => Dist.dist (↑e z) z)
    R : IsometryEquiv PE PE := (AffineIsometryEquiv.pointReflection Real z).toIsom …
    f : IsometryEquiv PE PE → IsometryEquiv PE PE := fun e => ((e.trans R).trans e …
    hf_dist : ∀ (e : IsometryEquiv PE PE), Eq (Dist.dist ((f e) z) z) (HMul.hMul 2 …
    hf_maps_to : Set.MapsTo f s s
    c : Real := iSup fun e => Dist.dist (↑e z) z
    this : LE.le c 0
    e : IsometryEquiv PE PE
    hx : Eq (e x) x
    hy : Eq (e y) y
    ⊢ LE.le (Dist.dist (e z) z) c
  -/
  exact le_ciSup h_bdd ⟨e, hx, hy⟩
  /-
    🎉 no goals
  -/


/-- A bijective isometry sends midpoints to midpoints. -/
theorem map_midpoint (f : PE ≃ᵢ PF) (x y : PE) : f (midpoint ℝ x y) = midpoint ℝ (f x) (f y) := by
  set e : PE ≃ᵢ PE :=
    ((f.trans <| (pointReflection ℝ <| midpoint ℝ (f x) (f y)).toIsometryEquiv).trans f.symm).trans
      (pointReflection ℝ <| midpoint ℝ x y).toIsometryEquiv
  /-
    E : Type u_1
    PE : Type u_2
    F : Type u_3
    PF : Type u_4
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : MetricSpace PE
    inst✝⁴ : NormedAddTorsor E PE
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : MetricSpace PF
    inst✝ : NormedAddTorsor F PF
    f : IsometryEquiv PE PF
    x y : PE
    e : IsometryEquiv PE PE := ((f.trans (AffineIsometryEquiv.pointReflection Real …
    ⊢ Eq (f (midpoint Real x y)) (midpoint Real (f x) (f y))
  -/
  have hx : e x = x := by simp [e]
  /-
    E : Type u_1
    PE : Type u_2
    F : Type u_3
    PF : Type u_4
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : MetricSpace PE
    inst✝⁴ : NormedAddTorsor E PE
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : MetricSpace PF
    inst✝ : NormedAddTorsor F PF
    f : IsometryEquiv PE PF
    x y : PE
    e : IsometryEquiv PE PE := ((f.trans (AffineIsometryEquiv.pointReflection Real …
    hx : Eq (e x) x
    ⊢ Eq (f (midpoint Real x y)) (midpoint Real (f x) (f y))
  -/
  have hy : e y = y := by simp [e]
  /-
    E : Type u_1
    PE : Type u_2
    F : Type u_3
    PF : Type u_4
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : MetricSpace PE
    inst✝⁴ : NormedAddTorsor E PE
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : MetricSpace PF
    inst✝ : NormedAddTorsor F PF
    f : IsometryEquiv PE PF
    x y : PE
    e : IsometryEquiv PE PE := ((f.trans (AffineIsometryEquiv.pointReflection Real …
    hx : Eq (e x) x
    hy : Eq (e y) y
    ⊢ Eq (f (midpoint Real x y)) (midpoint Real (f x) (f y))
  -/
  have hm := e.midpoint_fixed hx hy
  /-
    E : Type u_1
    PE : Type u_2
    F : Type u_3
    PF : Type u_4
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : MetricSpace PE
    inst✝⁴ : NormedAddTorsor E PE
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : MetricSpace PF
    inst✝ : NormedAddTorsor F PF
    f : IsometryEquiv PE PF
    x y : PE
    e : IsometryEquiv PE PE := ((f.trans (AffineIsometryEquiv.pointReflection Real …
    hx : Eq (e x) x
    hy : Eq (e y) y
    hm : Eq (e (midpoint Real x y)) (midpoint Real x y)
    ⊢ Eq (f (midpoint Real x y)) (midpoint Real (f x) (f y))
  -/
  simp only [e, trans_apply] at hm
  rwa [← eq_symm_apply, toIsometryEquiv_symm, pointReflection_symm, coe_toIsometryEquiv,
    coe_toIsometryEquiv, pointReflection_self, symm_apply_eq, @pointReflection_fixed_iff] at hm


/-- **Mazur-Ulam Theorem**: if `f` is an isometric bijection between two normed vector spaces
over `ℝ` and `f 0 = 0`, then `f` is a linear isometry equivalence. -/
def toRealLinearIsometryEquivOfMapZero (f : E ≃ᵢ F) (h0 : f 0 = 0) : E ≃ₗᵢ[ℝ] F :=
  { (AddMonoidHom.ofMapMidpoint ℝ ℝ f h0 f.map_midpoint).toRealLinearMap f.continuous, f with
                                              /-
                                                E : Type u_1
                                                PE : Type u_2
                                                F : Type u_3
                                                PF : Type u_4
                                                inst✝⁷ : NormedAddCommGroup E
                                                inst✝⁶ : NormedSpace Real E
                                                inst✝⁵ : MetricSpace PE
                                                inst✝⁴ : NormedAddTorsor E PE
                                                inst✝³ : NormedAddCommGroup F
                                                inst✝² : NormedSpace Real F
                                                inst✝¹ : MetricSpace PF
                                                inst✝ : NormedAddTorsor F PF
                                                f : IsometryEquiv E F
                                                h0 : Eq (f 0) 0
                                                x : E
                                                ⊢ Eq (Norm.norm (f x)) (Norm.norm x)
                                              -/
    norm_map' := fun x => show ‖f x‖ = ‖x‖ by simp only [← dist_zero_right, ← h0, f.dist_eq] }
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem coe_toRealLinearIsometryEquivOfMapZero (f : E ≃ᵢ F) (h0 : f 0 = 0) :
    ⇑(f.toRealLinearIsometryEquivOfMapZero h0) = f :=
  rfl


@[simp]
theorem coe_toRealLinearIsometryEquivOfMapZero_symm (f : E ≃ᵢ F) (h0 : f 0 = 0) :
    ⇑(f.toRealLinearIsometryEquivOfMapZero h0).symm = f.symm :=
  rfl


/-- **Mazur-Ulam Theorem**: if `f` is an isometric bijection between two normed vector spaces
over `ℝ`, then `x ↦ f x - f 0` is a linear isometry equivalence. -/
def toRealLinearIsometryEquiv (f : E ≃ᵢ F) : E ≃ₗᵢ[ℝ] F :=
  (f.trans (IsometryEquiv.addRight (f 0)).symm).toRealLinearIsometryEquivOfMapZero
        /-
          E : Type u_1
          PE : Type u_2
          F : Type u_3
          PF : Type u_4
          inst✝⁷ : NormedAddCommGroup E
          inst✝⁶ : NormedSpace Real E
          inst✝⁵ : MetricSpace PE
          inst✝⁴ : NormedAddTorsor E PE
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace Real F
          inst✝¹ : MetricSpace PF
          inst✝ : NormedAddTorsor F PF
          f : IsometryEquiv E F
          ⊢ Eq ((f.trans (IsometryEquiv.addRight (f 0)).symm) 0) 0
        -/
    (by simpa only [sub_eq_add_neg] using sub_self (f 0))
        /-
          🎉 no goals
        -/


@[simp]
theorem toRealLinearIsometryEquiv_apply (f : E ≃ᵢ F) (x : E) :
    (f.toRealLinearIsometryEquiv : E → F) x = f x - f 0 :=
  (sub_eq_add_neg (f x) (f 0)).symm


@[simp]
theorem toRealLinearIsometryEquiv_symm_apply (f : E ≃ᵢ F) (y : F) :
    (f.toRealLinearIsometryEquiv.symm : F → E) y = f.symm (y + f 0) :=
  rfl


/-- **Mazur-Ulam Theorem**: if `f` is an isometric bijection between two normed add-torsors over
normed vector spaces over `ℝ`, then `f` is an affine isometry equivalence. -/
def toRealAffineIsometryEquiv (f : PE ≃ᵢ PF) : PE ≃ᵃⁱ[ℝ] PF :=
  AffineIsometryEquiv.mk' f
    ((vaddConst (Classical.arbitrary PE)).trans <|
        f.trans (vaddConst (f <| Classical.arbitrary PE)).symm).toRealLinearIsometryEquiv
                                         /-
                                           E : Type u_1
                                           PE : Type u_2
                                           F : Type u_3
                                           PF : Type u_4
                                           inst✝⁷ : NormedAddCommGroup E
                                           inst✝⁶ : NormedSpace Real E
                                           inst✝⁵ : MetricSpace PE
                                           inst✝⁴ : NormedAddTorsor E PE
                                           inst✝³ : NormedAddCommGroup F
                                           inst✝² : NormedSpace Real F
                                           inst✝¹ : MetricSpace PF
                                           inst✝ : NormedAddTorsor F PF
                                           f : IsometryEquiv PE PF
                                           p : PE
                                           ⊢ Eq (f p) (HVAdd.hVAdd (((IsometryEquiv.vaddConst (Classical.arbitrary PE)).t …
                                         -/
    (Classical.arbitrary PE) fun p => by simp
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem coeFn_toRealAffineIsometryEquiv (f : PE ≃ᵢ PF) : ⇑f.toRealAffineIsometryEquiv = f :=
  rfl


@[simp]
theorem coe_toRealAffineIsometryEquiv (f : PE ≃ᵢ PF) :
    f.toRealAffineIsometryEquiv.toIsometryEquiv = f := by
  /-
    E : Type u_1
    PE : Type u_2
    F : Type u_3
    PF : Type u_4
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : MetricSpace PE
    inst✝⁴ : NormedAddTorsor E PE
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : MetricSpace PF
    inst✝ : NormedAddTorsor F PF
    f : IsometryEquiv PE PF
    ⊢ Eq f.toRealAffineIsometryEquiv.toIsometryEquiv f
  -/
  ext
  /-
    case H
    E : Type u_1
    PE : Type u_2
    F : Type u_3
    PF : Type u_4
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : MetricSpace PE
    inst✝⁴ : NormedAddTorsor E PE
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : MetricSpace PF
    inst✝ : NormedAddTorsor F PF
    f : IsometryEquiv PE PF
    x✝ : PE
    ⊢ Eq (f.toRealAffineIsometryEquiv.toIsometryEquiv x✝) (f x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


