/-- The frontier of a closed strictly convex set only contains trivial arithmetic progressions.
The idea is that an arithmetic progression is contained on a line and the frontier of a strictly
convex set does not contain lines. -/
lemma threeAPFree_frontier {𝕜 E : Type*} [LinearOrderedField 𝕜] [TopologicalSpace E]
    [AddCommMonoid E] [Module 𝕜 E] {s : Set E} (hs₀ : IsClosed s) (hs₁ : StrictConvex 𝕜 s) :
    ThreeAPFree (frontier s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    hs₀ : IsClosed s
    hs₁ : StrictConvex 𝕜 s
    ⊢ ThreeAPFree (frontier s)
  -/
  intro a ha b hb c hc habc
  obtain rfl : (1 / 2 : 𝕜) • a + (1 / 2 : 𝕜) • c = b := by
    rwa [← smul_add, one_div, inv_smul_eq_iff₀ (show (2 : 𝕜) ≠ 0 by norm_num), two_smul]
  have :=
    hs₁.eq (hs₀.frontier_subset ha) (hs₀.frontier_subset hc) one_half_pos one_half_pos
      (add_halves _) hb.2
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    hs₀ : IsClosed s
    hs₁ : StrictConvex 𝕜 s
    a : E
    ha : Membership.mem (frontier s) a
    c : E
    hc : Membership.mem (frontier s) c
    hb : Membership.mem (frontier s) (HAdd.hAdd (HSMul.hSMul (1 / 2) a) (HSMul.hSM …
    habc : Eq (HAdd.hAdd a c) (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul (1 / 2) a) (HSMul …
    this : Eq a c
    ⊢ Eq a (HAdd.hAdd (HSMul.hSMul (1 / 2) a) (HSMul.hSMul (1 / 2) c))
  -/
  simp [this, ← add_smul]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    hs₀ : IsClosed s
    hs₁ : StrictConvex 𝕜 s
    a : E
    ha : Membership.mem (frontier s) a
    c : E
    hc : Membership.mem (frontier s) c
    hb : Membership.mem (frontier s) (HAdd.hAdd (HSMul.hSMul (1 / 2) a) (HSMul.hSM …
    habc : Eq (HAdd.hAdd a c) (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul (1 / 2) a) (HSMul …
    this : Eq a c
    ⊢ Eq c (HSMul.hSMul (HAdd.hAdd (Inv.inv 2) (Inv.inv 2)) c)
  -/
  ring_nf
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    hs₀ : IsClosed s
    hs₁ : StrictConvex 𝕜 s
    a : E
    ha : Membership.mem (frontier s) a
    c : E
    hc : Membership.mem (frontier s) c
    hb : Membership.mem (frontier s) (HAdd.hAdd (HSMul.hSMul (1 / 2) a) (HSMul.hSM …
    habc : Eq (HAdd.hAdd a c) (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul (1 / 2) a) (HSMul …
    this : Eq a c
    ⊢ Eq c (HSMul.hSMul 1 c)
  -/
  simp
  /-
    🎉 no goals
  -/


lemma threeAPFree_sphere {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [StrictConvexSpace ℝ E] (x : E) (r : ℝ) : ThreeAPFree (sphere x r) := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : StrictConvexSpace Real E
    x : E
    r : Real
    ⊢ ThreeAPFree (Metric.sphere x r)
  -/
  obtain rfl | hr := eq_or_ne r 0
    /-
      case inl
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : StrictConvexSpace Real E
      x : E
      ⊢ ThreeAPFree (Metric.sphere x 0)
    -/
  · rw [sphere_zero]
    /-
      case inl
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : StrictConvexSpace Real E
      x : E
      ⊢ ThreeAPFree (Singleton.singleton x)
    -/
    exact threeAPFree_singleton _
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : StrictConvexSpace Real E
      x : E
      r : Real
      hr : Ne r 0
      ⊢ ThreeAPFree (Metric.sphere x r)
    -/
  · convert threeAPFree_frontier isClosed_ball (strictConvex_closedBall ℝ x r)
    /-
      case h.e'_3
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : StrictConvexSpace Real E
      x : E
      r : Real
      hr : Ne r 0
      ⊢ Eq (Metric.sphere x r) (frontier (Metric.closedBall x r))
    -/
    exact (frontier_closedBall _ hr).symm
    /-
      🎉 no goals
    -/


/-- The box `{0, ..., d - 1}^n` as a `Finset`. -/
def box (n d : ℕ) : Finset (Fin n → ℕ) :=
  Fintype.piFinset fun _ => range d


                                                   /-
                                                     n d : Nat
                                                     x : Fin n → Nat
                                                     ⊢ Iff (Membership.mem (Behrend.box n d) x) (∀ (i : Fin n), LT.lt (x i) d)
                                                   -/
theorem mem_box : x ∈ box n d ↔ ∀ i, x i < d := by simp only [box, Fintype.mem_piFinset, mem_range]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
                                            /-
                                              n d : Nat
                                              ⊢ Eq (Behrend.box n d).card (HPow.hPow d n)
                                            -/
theorem card_box : #(box n d) = d ^ n := by simp [box]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
                                           /-
                                             n : Nat
                                             ⊢ Eq (Behrend.box (HAdd.hAdd n 1) 0) EmptyCollection.emptyCollection
                                           -/
theorem box_zero : box (n + 1) 0 = ∅ := by simp [box]
                                           /-
                                             🎉 no goals
                                           -/


/-- The intersection of the sphere of radius `√k` with the integer points in the positive
quadrant. -/
def sphere (n d k : ℕ) : Finset (Fin n → ℕ) := {x ∈ box n d | ∑ i, x i ^ 2 = k}


                                                             /-
                                                               n d : Nat
                                                               x : Fin n → Nat
                                                               ⊢ Membership.mem (Behrend.sphere n d 0) x → Membership.mem 0 x
                                                             -/
theorem sphere_zero_subset : sphere n d 0 ⊆ 0 := fun x => by simp [sphere, funext_iff]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
                                                                   /-
                                                                     n k : Nat
                                                                     ⊢ Eq (Behrend.sphere (HAdd.hAdd n 1) 0 k) EmptyCollection.emptyCollection
                                                                   -/
theorem sphere_zero_right (n k : ℕ) : sphere (n + 1) 0 k = ∅ := by simp [sphere]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem sphere_subset_box : sphere n d k ⊆ box n d :=
  filter_subset _ _


theorem norm_of_mem_sphere {x : Fin n → ℕ} (hx : x ∈ sphere n d k) :
    ‖(WithLp.equiv 2 _).symm ((↑) ∘ x : Fin n → ℝ)‖ = √↑k := by
  /-
    n d k : Nat
    x : Fin n → Nat
    hx : Membership.mem (Behrend.sphere n d k) x
    ⊢ Eq (Norm.norm ((WithLp.equiv 2 (Fin n → Real)).symm (Function.comp Nat.cast  …
  -/
  rw [EuclideanSpace.norm_eq]
  /-
    n d k : Nat
    x : Fin n → Nat
    hx : Membership.mem (Behrend.sphere n d k) x
    ⊢ Eq (Finset.univ.sum fun i => HPow.hPow (Norm.norm ((WithLp.equiv 2 (Fin n →  …
  -/
  dsimp
  /-
    n d k : Nat
    x : Fin n → Nat
    hx : Membership.mem (Behrend.sphere n d k) x
    ⊢ Eq (Finset.univ.sum fun i => HPow.hPow (abs ↑(x i)) 2).sqrt (↑k).sqrt
  -/
  simp_rw [abs_cast, ← cast_pow, ← cast_sum, (mem_filter.1 hx).2]
  /-
    🎉 no goals
  -/


theorem sphere_subset_preimage_metric_sphere : (sphere n d k : Set (Fin n → ℕ)) ⊆
    (fun x : Fin n → ℕ => (WithLp.equiv 2 _).symm ((↑) ∘ x : Fin n → ℝ)) ⁻¹'
      Metric.sphere (0 : PiLp 2 fun _ : Fin n => ℝ) (√↑k) :=
                 /-
                   n d k : Nat
                   x : Fin n → Nat
                   hx : Membership.mem (↑(Behrend.sphere n d k)) x
                   ⊢ Membership.mem (Set.preimage (fun x => (WithLp.equiv 2 (Fin n → Real)).symm  …
                 -/
  fun x hx => by rw [Set.mem_preimage, mem_sphere_zero_iff_norm, norm_of_mem_sphere hx]
                 /-
                   🎉 no goals
                 -/


/-- The map that appears in Behrend's bound on Roth numbers. -/
@[simps]
def map (d : ℕ) : (Fin n → ℕ) →+ ℕ where
  toFun a := ∑ i, a i * d ^ (i : ℕ)
                  /-
                    n d✝ k N : Nat
                    x : Fin n → Nat
                    d : Nat
                    ⊢ Eq ((fun a => Finset.univ.sum fun i => HMul.hMul (a i) (HPow.hPow d ↑i)) 0) 0
                  -/
  map_zero' := by simp_rw [Pi.zero_apply, zero_mul, sum_const_zero]
                  /-
                    🎉 no goals
                  -/
                     /-
                       n d✝ k N : Nat
                       x : Fin n → Nat
                       d : Nat
                       a b : Fin n → Nat
                       ⊢ Eq ({ toFun := fun a => Finset.univ.sum fun i => HMul.hMul (a i) (HPow.hPow  …
                     -/
  map_add' a b := by simp_rw [Pi.add_apply, add_mul, sum_add_distrib]
                     /-
                       🎉 no goals
                     -/


                                                             /-
                                                               d : Nat
                                                               a : Fin 0 → Nat
                                                               ⊢ Eq ((Behrend.map d) a) 0
                                                             -/
theorem map_zero (d : ℕ) (a : Fin 0 → ℕ) : map d a = 0 := by simp [map]
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem map_succ (a : Fin (n + 1) → ℕ) :
    map d a = a 0 + (∑ x : Fin n, a x.succ * d ^ (x : ℕ)) * d := by
  /-
    n d : Nat
    a : Fin (HAdd.hAdd n 1) → Nat
    ⊢ Eq ((Behrend.map d) a) (HAdd.hAdd (a 0) (HMul.hMul (Finset.univ.sum fun x => …
  -/
  simp [map, Fin.sum_univ_succ, _root_.pow_succ, ← mul_assoc, ← sum_mul]
  /-
    🎉 no goals
  -/


theorem map_succ' (a : Fin (n + 1) → ℕ) : map d a = a 0 + map d (a ∘ Fin.succ) * d :=
  map_succ _


theorem map_monotone (d : ℕ) : Monotone (map d : (Fin n → ℕ) → ℕ) := fun x y h => by
  /-
    n d : Nat
    x y : Fin n → Nat
    h : LE.le x y
    ⊢ LE.le ((Behrend.map d) x) ((Behrend.map d) y)
  -/
  dsimp; exact sum_le_sum fun i _ => Nat.mul_le_mul_right _ <| h i
         /-
           🎉 no goals
         -/


theorem map_mod (a : Fin n.succ → ℕ) : map d a % d = a 0 % d := by
  /-
    n d : Nat
    a : Fin n.succ → Nat
    ⊢ Eq (HMod.hMod ((Behrend.map d) a) d) (HMod.hMod (a 0) d)
  -/
  rw [map_succ, Nat.add_mul_mod_self_right]
  /-
    🎉 no goals
  -/


theorem map_eq_iff {x₁ x₂ : Fin n.succ → ℕ} (hx₁ : ∀ i, x₁ i < d) (hx₂ : ∀ i, x₂ i < d) :
    map d x₁ = map d x₂ ↔ x₁ 0 = x₂ 0 ∧ map d (x₁ ∘ Fin.succ) = map d (x₂ ∘ Fin.succ) := by
  /-
    n d : Nat
    x₁ x₂ : Fin n.succ → Nat
    hx₁ : ∀ (i : Fin n.succ), LT.lt (x₁ i) d
    hx₂ : ∀ (i : Fin n.succ), LT.lt (x₂ i) d
    ⊢ Iff (Eq ((Behrend.map d) x₁) ((Behrend.map d) x₂)) (And (Eq (x₁ 0) (x₂ 0)) ( …
  -/
  refine ⟨fun h => ?_, fun h => by rw [map_succ', map_succ', h.1, h.2]⟩
  have : x₁ 0 = x₂ 0 := by
    rw [← mod_eq_of_lt (hx₁ _), ← map_mod, ← mod_eq_of_lt (hx₂ _), ← map_mod, h]
  /-
    n d : Nat
    x₁ x₂ : Fin n.succ → Nat
    hx₁ : ∀ (i : Fin n.succ), LT.lt (x₁ i) d
    hx₂ : ∀ (i : Fin n.succ), LT.lt (x₂ i) d
    h : Eq ((Behrend.map d) x₁) ((Behrend.map d) x₂)
    this : Eq (x₁ 0) (x₂ 0)
    ⊢ And (Eq (x₁ 0) (x₂ 0)) (Eq ((Behrend.map d) (Function.comp x₁ Fin.succ)) ((B …
  -/
  rw [map_succ, map_succ, this, add_right_inj, mul_eq_mul_right_iff] at h
  /-
    n d : Nat
    x₁ x₂ : Fin n.succ → Nat
    hx₁ : ∀ (i : Fin n.succ), LT.lt (x₁ i) d
    hx₂ : ∀ (i : Fin n.succ), LT.lt (x₂ i) d
    h : Or (Eq (Finset.univ.sum fun x => HMul.hMul (x₁ x.succ) (HPow.hPow d ↑x)) ( …
    this : Eq (x₁ 0) (x₂ 0)
    ⊢ And (Eq (x₁ 0) (x₂ 0)) (Eq ((Behrend.map d) (Function.comp x₁ Fin.succ)) ((B …
  -/
  exact ⟨this, h.resolve_right (pos_of_gt (hx₁ 0)).ne'⟩
  /-
    🎉 no goals
  -/


theorem map_injOn : {x : Fin n → ℕ | ∀ i, x i < d}.InjOn (map d) := by
  /-
    n d : Nat
    ⊢ Set.InjOn (⇑(Behrend.map d)) (setOf fun x => ∀ (i : Fin n), LT.lt (x i) d)
  -/
  intro x₁ hx₁ x₂ hx₂ h
  /-
    n d : Nat
    x₁ : Fin n → Nat
    hx₁ : Membership.mem (setOf fun x => ∀ (i : Fin n), LT.lt (x i) d) x₁
    x₂ : Fin n → Nat
    hx₂ : Membership.mem (setOf fun x => ∀ (i : Fin n), LT.lt (x i) d) x₂
    h : Eq ((Behrend.map d) x₁) ((Behrend.map d) x₂)
    ⊢ Eq x₁ x₂
  -/
  induction' n with n ih
    /-
      case zero
      n d : Nat
      x₁ : Fin 0 → Nat
      hx₁ : Membership.mem (setOf fun x => ∀ (i : Fin 0), LT.lt (x i) d) x₁
      x₂ : Fin 0 → Nat
      hx₂ : Membership.mem (setOf fun x => ∀ (i : Fin 0), LT.lt (x i) d) x₂
      h : Eq ((Behrend.map d) x₁) ((Behrend.map d) x₂)
      ⊢ Eq x₁ x₂
    -/
  · simp [eq_iff_true_of_subsingleton]
    /-
      🎉 no goals
    -/
  /-
    case succ
    n✝ d n : Nat
    ih : ∀ ⦃x₁ : Fin n → Nat⦄, Membership.mem (setOf fun x => ∀ (i : Fin n), LT.lt …
    x₁ : Fin (HAdd.hAdd n 1) → Nat
    hx₁ : Membership.mem (setOf fun x => ∀ (i : Fin (HAdd.hAdd n 1)), LT.lt (x i)  …
    x₂ : Fin (HAdd.hAdd n 1) → Nat
    hx₂ : Membership.mem (setOf fun x => ∀ (i : Fin (HAdd.hAdd n 1)), LT.lt (x i)  …
    h : Eq ((Behrend.map d) x₁) ((Behrend.map d) x₂)
    ⊢ Eq x₁ x₂
  -/
  ext i
  /-
    case succ.h
    n✝ d n : Nat
    ih : ∀ ⦃x₁ : Fin n → Nat⦄, Membership.mem (setOf fun x => ∀ (i : Fin n), LT.lt …
    x₁ : Fin (HAdd.hAdd n 1) → Nat
    hx₁ : Membership.mem (setOf fun x => ∀ (i : Fin (HAdd.hAdd n 1)), LT.lt (x i)  …
    x₂ : Fin (HAdd.hAdd n 1) → Nat
    hx₂ : Membership.mem (setOf fun x => ∀ (i : Fin (HAdd.hAdd n 1)), LT.lt (x i)  …
    h : Eq ((Behrend.map d) x₁) ((Behrend.map d) x₂)
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (x₁ i) (x₂ i)
  -/
  have x := (map_eq_iff hx₁ hx₂).1 h
  /-
    case succ.h
    n✝ d n : Nat
    ih : ∀ ⦃x₁ : Fin n → Nat⦄, Membership.mem (setOf fun x => ∀ (i : Fin n), LT.lt …
    x₁ : Fin (HAdd.hAdd n 1) → Nat
    hx₁ : Membership.mem (setOf fun x => ∀ (i : Fin (HAdd.hAdd n 1)), LT.lt (x i)  …
    x₂ : Fin (HAdd.hAdd n 1) → Nat
    hx₂ : Membership.mem (setOf fun x => ∀ (i : Fin (HAdd.hAdd n 1)), LT.lt (x i)  …
    h : Eq ((Behrend.map d) x₁) ((Behrend.map d) x₂)
    i : Fin (HAdd.hAdd n 1)
    x : And (Eq (x₁ 0) (x₂ 0)) (Eq ((Behrend.map d) (Function.comp x₁ Fin.succ)) ( …
    ⊢ Eq (x₁ i) (x₂ i)
  -/
  refine Fin.cases x.1 (congr_fun <| ih (fun _ => ?_) (fun _ => ?_) x.2) i
    /-
      case succ.h.refine_1
      n✝ d n : Nat
      ih : ∀ ⦃x₁ : Fin n → Nat⦄, Membership.mem (setOf fun x => ∀ (i : Fin n), LT.lt …
      x₁ : Fin (HAdd.hAdd n 1) → Nat
      hx₁ : Membership.mem (setOf fun x => ∀ (i : Fin (HAdd.hAdd n 1)), LT.lt (x i)  …
      x₂ : Fin (HAdd.hAdd n 1) → Nat
      hx₂ : Membership.mem (setOf fun x => ∀ (i : Fin (HAdd.hAdd n 1)), LT.lt (x i)  …
      h : Eq ((Behrend.map d) x₁) ((Behrend.map d) x₂)
      i : Fin (HAdd.hAdd n 1)
      x : And (Eq (x₁ 0) (x₂ 0)) (Eq ((Behrend.map d) (Function.comp x₁ Fin.succ)) ( …
      x✝ : Fin n
      ⊢ LT.lt (x₁ x✝.succ) d
    -/
  · exact hx₁ _
    /-
      🎉 no goals
    -/
    /-
      case succ.h.refine_2
      n✝ d n : Nat
      ih : ∀ ⦃x₁ : Fin n → Nat⦄, Membership.mem (setOf fun x => ∀ (i : Fin n), LT.lt …
      x₁ : Fin (HAdd.hAdd n 1) → Nat
      hx₁ : Membership.mem (setOf fun x => ∀ (i : Fin (HAdd.hAdd n 1)), LT.lt (x i)  …
      x₂ : Fin (HAdd.hAdd n 1) → Nat
      hx₂ : Membership.mem (setOf fun x => ∀ (i : Fin (HAdd.hAdd n 1)), LT.lt (x i)  …
      h : Eq ((Behrend.map d) x₁) ((Behrend.map d) x₂)
      i : Fin (HAdd.hAdd n 1)
      x : And (Eq (x₁ 0) (x₂ 0)) (Eq ((Behrend.map d) (Function.comp x₁ Fin.succ)) ( …
      x✝ : Fin n
      ⊢ LT.lt (x₂ x✝.succ) d
    -/
  · exact hx₂ _
    /-
      🎉 no goals
    -/


theorem map_le_of_mem_box (hx : x ∈ box n d) :
    map (2 * d - 1) x ≤ ∑ i : Fin n, (d - 1) * (2 * d - 1) ^ (i : ℕ) :=
  map_monotone (2 * d - 1) fun _ => Nat.le_sub_one_of_lt <| mem_box.1 hx _


nonrec theorem threeAPFree_sphere : ThreeAPFree (sphere n d k : Set (Fin n → ℕ)) := by
  set f : (Fin n → ℕ) →+ EuclideanSpace ℝ (Fin n) :=
    { toFun := fun f => ((↑) : ℕ → ℝ) ∘ f
      map_zero' := funext fun _ => cast_zero
      map_add' := fun _ _ => funext fun _ => cast_add _ _ }
  refine ThreeAPFree.of_image (AddMonoidHomClass.isAddFreimanHom f (Set.mapsTo_image _ _))
    cast_injective.comp_left.injOn (Set.subset_univ _) ?_
  /-
    n d k : Nat
    f : AddMonoidHom (Fin n → Nat) (EuclideanSpace Real (Fin n)) := { toFun := fun …
    ⊢ ThreeAPFree (Set.image ⇑f ↑(Behrend.sphere n d k))
  -/
  refine (threeAPFree_sphere 0 (√↑k)).mono (Set.image_subset_iff.2 fun x => ?_)
  /-
    n d k : Nat
    f : AddMonoidHom (Fin n → Nat) (EuclideanSpace Real (Fin n)) := { toFun := fun …
    x : Fin n → Nat
    ⊢ Membership.mem (↑(Behrend.sphere n d k)) x → Membership.mem (Set.preimage (⇑ …
  -/
  rw [Set.mem_preimage, mem_sphere_zero_iff_norm]
  /-
    n d k : Nat
    f : AddMonoidHom (Fin n → Nat) (EuclideanSpace Real (Fin n)) := { toFun := fun …
    x : Fin n → Nat
    ⊢ Membership.mem (↑(Behrend.sphere n d k)) x → Eq (Norm.norm (f x)) (↑k).sqrt
  -/
  exact norm_of_mem_sphere
  /-
    🎉 no goals
  -/


theorem threeAPFree_image_sphere :
    ThreeAPFree ((sphere n d k).image (map (2 * d - 1)) : Set ℕ) := by
  /-
    n d k : Nat
    ⊢ ThreeAPFree ↑(Finset.image (⇑(Behrend.map (HSub.hSub (HMul.hMul 2 d) 1))) (B …
  -/
  rw [coe_image]
  apply ThreeAPFree.image' (α := Fin n → ℕ) (β := ℕ) (s := sphere n d k) (map (2 * d - 1))
    (map_injOn.mono _) threeAPFree_sphere
  /-
    n d k : Nat
    ⊢ HasSubset.Subset (HAdd.hAdd ↑(Behrend.sphere n d k) ↑(Behrend.sphere n d k)) …
  -/
  rw [Set.add_subset_iff]
  /-
    n d k : Nat
    ⊢ ∀ (x : Fin n → Nat), Membership.mem (↑(Behrend.sphere n d k)) x → ∀ (y : Fin …
  -/
  rintro a ha b hb i
  /-
    n d k : Nat
    a : Fin n → Nat
    ha : Membership.mem (↑(Behrend.sphere n d k)) a
    b : Fin n → Nat
    hb : Membership.mem (↑(Behrend.sphere n d k)) b
    i : Fin n
    ⊢ LT.lt (HAdd.hAdd a b i) (HSub.hSub (HMul.hMul 2 d) 1)
  -/
  have hai := mem_box.1 (sphere_subset_box ha) i
  /-
    n d k : Nat
    a : Fin n → Nat
    ha : Membership.mem (↑(Behrend.sphere n d k)) a
    b : Fin n → Nat
    hb : Membership.mem (↑(Behrend.sphere n d k)) b
    i : Fin n
    hai : LT.lt (a i) d
    ⊢ LT.lt (HAdd.hAdd a b i) (HSub.hSub (HMul.hMul 2 d) 1)
  -/
  have hbi := mem_box.1 (sphere_subset_box hb) i
  /-
    n d k : Nat
    a : Fin n → Nat
    ha : Membership.mem (↑(Behrend.sphere n d k)) a
    b : Fin n → Nat
    hb : Membership.mem (↑(Behrend.sphere n d k)) b
    i : Fin n
    hai : LT.lt (a i) d
    hbi : LT.lt (b i) d
    ⊢ LT.lt (HAdd.hAdd a b i) (HSub.hSub (HMul.hMul 2 d) 1)
  -/
  rw [lt_tsub_iff_right, ← succ_le_iff, two_mul]
  /-
    n d k : Nat
    a : Fin n → Nat
    ha : Membership.mem (↑(Behrend.sphere n d k)) a
    b : Fin n → Nat
    hb : Membership.mem (↑(Behrend.sphere n d k)) b
    i : Fin n
    hai : LT.lt (a i) d
    hbi : LT.lt (b i) d
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd a b i) 1).succ (HAdd.hAdd d d)
  -/
  exact (add_add_add_comm _ _ 1 1).trans_le (_root_.add_le_add hai hbi)
  /-
    🎉 no goals
  -/


theorem sum_sq_le_of_mem_box (hx : x ∈ box n d) : ∑ i : Fin n, x i ^ 2 ≤ n * (d - 1) ^ 2 := by
  /-
    n d : Nat
    x : Fin n → Nat
    hx : Membership.mem (Behrend.box n d) x
    ⊢ LE.le (Finset.univ.sum fun i => HPow.hPow (x i) 2) (HMul.hMul n (HPow.hPow ( …
  -/
  rw [mem_box] at hx
  have : ∀ i, x i ^ 2 ≤ (d - 1) ^ 2 := fun i =>
    Nat.pow_le_pow_left (Nat.le_sub_one_of_lt (hx i)) _
  /-
    n d : Nat
    x : Fin n → Nat
    hx : ∀ (i : Fin n), LT.lt (x i) d
    this : ∀ (i : Fin n), LE.le (HPow.hPow (x i) 2) (HPow.hPow (HSub.hSub d 1) 2)
    ⊢ LE.le (Finset.univ.sum fun i => HPow.hPow (x i) 2) (HMul.hMul n (HPow.hPow ( …
  -/
  exact (sum_le_card_nsmul univ _ _ fun i _ => this i).trans (by rw [card_fin, smul_eq_mul])
  /-
    🎉 no goals
  -/


theorem sum_eq : (∑ i : Fin n, d * (2 * d + 1) ^ (i : ℕ)) = ((2 * d + 1) ^ n - 1) / 2 := by
  /-
    n d : Nat
    ⊢ Eq (Finset.univ.sum fun i => HMul.hMul d (HPow.hPow (HAdd.hAdd (HMul.hMul 2  …
  -/
  refine (Nat.div_eq_of_eq_mul_left zero_lt_two ?_).symm
  rw [← sum_range fun i => d * (2 * d + 1) ^ (i : ℕ), ← mul_sum, mul_right_comm, mul_comm d, ←
    geom_sum_mul_add, add_tsub_cancel_right, mul_comm]


theorem sum_lt : (∑ i : Fin n, d * (2 * d + 1) ^ (i : ℕ)) < (2 * d + 1) ^ n :=
  sum_eq.trans_lt <| (Nat.div_le_self _ 2).trans_lt <| pred_lt (pow_pos (succ_pos _) _).ne'


theorem card_sphere_le_rothNumberNat (n d k : ℕ) :
    #(sphere n d k) ≤ rothNumberNat ((2 * d - 1) ^ n) := by
  /-
    n d k : Nat
    ⊢ LE.le (Behrend.sphere n d k).card (rothNumberNat (HPow.hPow (HSub.hSub (HMul …
  -/
  cases n
    /-
      case zero
      d k : Nat
      ⊢ LE.le (Behrend.sphere 0 d k).card (rothNumberNat (HPow.hPow (HSub.hSub (HMul …
    -/
  · dsimp; refine (card_le_univ _).trans_eq ?_; rfl
                                                /-
                                                  🎉 no goals
                                                -/
  /-
    case succ
    d k n✝ : Nat
    ⊢ LE.le (Behrend.sphere (HAdd.hAdd n✝ 1) d k).card (rothNumberNat (HPow.hPow ( …
  -/
  cases d
    /-
      case succ.zero
      k n✝ : Nat
      ⊢ LE.le (Behrend.sphere (HAdd.hAdd n✝ 1) 0 k).card (rothNumberNat (HPow.hPow ( …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ.succ
    k n✝¹ n✝ : Nat
    ⊢ LE.le (Behrend.sphere (HAdd.hAdd n✝¹ 1) (HAdd.hAdd n✝ 1) k).card (rothNumber …
  -/
  apply threeAPFree_image_sphere.le_rothNumberNat _ _ (card_image_of_injOn _)
  · simp only [subset_iff, mem_image, and_imp, forall_exists_index, mem_range,
      forall_apply_eq_imp_iff₂, sphere, mem_filter]
    /-
      k n✝¹ n✝ : Nat
      ⊢ ∀ (x : Nat) (x_1 : Fin (HAdd.hAdd n✝¹ 1) → Nat), Membership.mem (Behrend.box …
    -/
    rintro _ x hx _ rfl
    /-
      k n✝¹ n✝ : Nat
      x : Fin (HAdd.hAdd n✝¹ 1) → Nat
      hx : Membership.mem (Behrend.box (HAdd.hAdd n✝¹ 1) (HAdd.hAdd n✝ 1)) x
      a✝ : Eq (Finset.univ.sum fun i => HPow.hPow (x i) 2) k
      ⊢ LT.lt ((Behrend.map (HSub.hSub (HMul.hMul 2 (HAdd.hAdd n✝ 1)) 1)) x) (HPow.h …
    -/
    exact (map_le_of_mem_box hx).trans_lt sum_lt
    /-
      🎉 no goals
    -/
  /-
    k n✝¹ n✝ : Nat
    ⊢ Set.InjOn ⇑(Behrend.map (HSub.hSub (HMul.hMul 2 (HAdd.hAdd n✝ 1)) 1)) ↑(Behr …
  -/
  apply map_injOn.mono fun x => ?_
  /-
    k n✝¹ n✝ : Nat
    x : Fin (HAdd.hAdd n✝¹ 1) → Nat
    ⊢ Membership.mem (↑(Behrend.sphere (HAdd.hAdd n✝¹ 1) (HAdd.hAdd n✝ 1) k)) x →  …
  -/
  simp only [mem_coe, sphere, mem_filter, mem_box, and_imp, two_mul]
  /-
    k n✝¹ n✝ : Nat
    x : Fin (HAdd.hAdd n✝¹ 1) → Nat
    ⊢ (∀ (i : Fin (HAdd.hAdd n✝¹ 1)), LT.lt (x i) (HAdd.hAdd n✝ 1)) → Eq (Finset.u …
  -/
  exact fun h _ i => (h i).trans_le le_self_add
  /-
    🎉 no goals
  -/


theorem exists_large_sphere_aux (n d : ℕ) : ∃ k ∈ range (n * (d - 1) ^ 2 + 1),
    (↑(d ^ n) / ((n * (d - 1) ^ 2 :) + 1) : ℝ) ≤ #(sphere n d k) := by
  /-
    n d : Nat
    ⊢ Exists fun k => And (Membership.mem (Finset.range (HAdd.hAdd (HMul.hMul n (H …
  -/
  refine exists_le_card_fiber_of_nsmul_le_card_of_maps_to (fun x hx => ?_) nonempty_range_succ ?_
    /-
      case refine_1
      n d : Nat
      x : Fin n → Nat
      hx : Membership.mem (Behrend.box n d) x
      ⊢ Membership.mem (Finset.range (HAdd.hAdd (HMul.hMul n (HPow.hPow (HSub.hSub d …
    -/
  · rw [mem_range, Nat.lt_succ_iff]
    /-
      case refine_1
      n d : Nat
      x : Fin n → Nat
      hx : Membership.mem (Behrend.box n d) x
      ⊢ LE.le (Finset.univ.sum fun i => HPow.hPow (x i) 2) (HMul.hMul n (HPow.hPow ( …
    -/
    exact sum_sq_le_of_mem_box hx
    /-
      🎉 no goals
    -/
  · rw [card_range, _root_.nsmul_eq_mul, mul_div_assoc', cast_add_one, mul_div_cancel_left₀,
      card_box]
    /-
      case refine_2.ha
      n d : Nat
      ⊢ Ne (HAdd.hAdd (↑(HMul.hMul n (HPow.hPow (HSub.hSub d 1) 2))) 1) 0
    -/
    exact (cast_add_one_pos _).ne'
    /-
      🎉 no goals
    -/


theorem exists_large_sphere (n d : ℕ) :
    ∃ k, ((d ^ n :) / (n * d ^ 2 :) : ℝ) ≤ #(sphere n d k) := by
  /-
    n d : Nat
    ⊢ Exists fun k => LE.le (HDiv.hDiv ↑(HPow.hPow d n) ↑(HMul.hMul n (HPow.hPow d …
  -/
  obtain ⟨k, -, hk⟩ := exists_large_sphere_aux n d
  /-
    case intro.intro
    n d k : Nat
    hk : LE.le (HDiv.hDiv (↑(HPow.hPow d n)) (HAdd.hAdd (↑(HMul.hMul n (HPow.hPow  …
    ⊢ Exists fun k => LE.le (HDiv.hDiv ↑(HPow.hPow d n) ↑(HMul.hMul n (HPow.hPow d …
  -/
  refine ⟨k, ?_⟩
  /-
    case intro.intro
    n d k : Nat
    hk : LE.le (HDiv.hDiv (↑(HPow.hPow d n)) (HAdd.hAdd (↑(HMul.hMul n (HPow.hPow  …
    ⊢ LE.le (HDiv.hDiv ↑(HPow.hPow d n) ↑(HMul.hMul n (HPow.hPow d 2))) ↑(Behrend. …
  -/
  obtain rfl | hn := n.eq_zero_or_pos
    /-
      case intro.intro.inl
      d k : Nat
      hk : LE.le (HDiv.hDiv (↑(HPow.hPow d 0)) (HAdd.hAdd (↑(HMul.hMul 0 (HPow.hPow  …
      ⊢ LE.le (HDiv.hDiv ↑(HPow.hPow d 0) ↑(HMul.hMul 0 (HPow.hPow d 2))) ↑(Behrend. …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.inr
    n d k : Nat
    hk : LE.le (HDiv.hDiv (↑(HPow.hPow d n)) (HAdd.hAdd (↑(HMul.hMul n (HPow.hPow  …
    hn : GT.gt n 0
    ⊢ LE.le (HDiv.hDiv ↑(HPow.hPow d n) ↑(HMul.hMul n (HPow.hPow d 2))) ↑(Behrend. …
  -/
  obtain rfl | hd := d.eq_zero_or_pos
    /-
      case intro.intro.inr.inl
      n k : Nat
      hn : GT.gt n 0
      hk : LE.le (HDiv.hDiv (↑(HPow.hPow 0 n)) (HAdd.hAdd (↑(HMul.hMul n (HPow.hPow  …
      ⊢ LE.le (HDiv.hDiv ↑(HPow.hPow 0 n) ↑(HMul.hMul n (HPow.hPow 0 2))) ↑(Behrend. …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.inr.inr
    n d k : Nat
    hk : LE.le (HDiv.hDiv (↑(HPow.hPow d n)) (HAdd.hAdd (↑(HMul.hMul n (HPow.hPow  …
    hn : GT.gt n 0
    hd : GT.gt d 0
    ⊢ LE.le (HDiv.hDiv ↑(HPow.hPow d n) ↑(HMul.hMul n (HPow.hPow d 2))) ↑(Behrend. …
  -/
  refine (div_le_div_of_nonneg_left ?_ ?_ ?_).trans hk
    /-
      case intro.intro.inr.inr.refine_1
      n d k : Nat
      hk : LE.le (HDiv.hDiv (↑(HPow.hPow d n)) (HAdd.hAdd (↑(HMul.hMul n (HPow.hPow  …
      hn : GT.gt n 0
      hd : GT.gt d 0
      ⊢ LE.le 0 ↑(HPow.hPow d n)
    -/
  · exact cast_nonneg _
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr.inr.refine_2
      n d k : Nat
      hk : LE.le (HDiv.hDiv (↑(HPow.hPow d n)) (HAdd.hAdd (↑(HMul.hMul n (HPow.hPow  …
      hn : GT.gt n 0
      hd : GT.gt d 0
      ⊢ LT.lt 0 (HAdd.hAdd (↑(HMul.hMul n (HPow.hPow (HSub.hSub d 1) 2))) 1)
    -/
  · exact cast_add_one_pos _
    /-
      🎉 no goals
    -/
  simp only [← le_sub_iff_add_le', cast_mul, ← mul_sub, cast_pow, cast_sub hd, sub_sq, one_pow,
    cast_one, mul_one, sub_add, sub_sub_self]
  /-
    case intro.intro.inr.inr.refine_3
    n d k : Nat
    hk : LE.le (HDiv.hDiv (↑(HPow.hPow d n)) (HAdd.hAdd (↑(HMul.hMul n (HPow.hPow  …
    hn : GT.gt n 0
    hd : GT.gt d 0
    ⊢ LE.le 1 (HMul.hMul (↑n) (HSub.hSub (HMul.hMul 2 ↑d) 1))
  -/
  apply one_le_mul_of_one_le_of_one_le
    /-
      case intro.intro.inr.inr.refine_3.ha
      n d k : Nat
      hk : LE.le (HDiv.hDiv (↑(HPow.hPow d n)) (HAdd.hAdd (↑(HMul.hMul n (HPow.hPow  …
      hn : GT.gt n 0
      hd : GT.gt d 0
      ⊢ LE.le 1 ↑n
    -/
  · rwa [one_le_cast]
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.inr.inr.refine_3.hb
    n d k : Nat
    hk : LE.le (HDiv.hDiv (↑(HPow.hPow d n)) (HAdd.hAdd (↑(HMul.hMul n (HPow.hPow  …
    hn : GT.gt n 0
    hd : GT.gt d 0
    ⊢ LE.le 1 (HSub.hSub (HMul.hMul 2 ↑d) 1)
  -/
  rw [_root_.le_sub_iff_add_le]
  /-
    case intro.intro.inr.inr.refine_3.hb
    n d k : Nat
    hk : LE.le (HDiv.hDiv (↑(HPow.hPow d n)) (HAdd.hAdd (↑(HMul.hMul n (HPow.hPow  …
    hn : GT.gt n 0
    hd : GT.gt d 0
    ⊢ LE.le (HAdd.hAdd 1 1) (HMul.hMul 2 ↑d)
  -/
  norm_num
  /-
    case intro.intro.inr.inr.refine_3.hb
    n d k : Nat
    hk : LE.le (HDiv.hDiv (↑(HPow.hPow d n)) (HAdd.hAdd (↑(HMul.hMul n (HPow.hPow  …
    hn : GT.gt n 0
    hd : GT.gt d 0
    ⊢ LE.le 1 d
  -/
  exact one_le_cast.2 hd
  /-
    🎉 no goals
  -/


theorem bound_aux' (n d : ℕ) : ((d ^ n :) / (n * d ^ 2 :) : ℝ) ≤ rothNumberNat ((2 * d - 1) ^ n) :=
  let ⟨_, h⟩ := exists_large_sphere n d
  h.trans <| cast_le.2 <| card_sphere_le_rothNumberNat _ _ _


theorem bound_aux (hd : d ≠ 0) (hn : 2 ≤ n) :
    (d ^ (n - 2 :) / n : ℝ) ≤ rothNumberNat ((2 * d - 1) ^ n) := by
  /-
    n d : Nat
    hd : Ne d 0
    hn : LE.le 2 n
    ⊢ LE.le (HDiv.hDiv (HPow.hPow (↑d) (HSub.hSub n 2)) ↑n) ↑(rothNumberNat (HPow. …
  -/
  convert bound_aux' n d using 1
  /-
    case h.e'_3
    n d : Nat
    hd : Ne d 0
    hn : LE.le 2 n
    ⊢ Eq (HDiv.hDiv (HPow.hPow (↑d) (HSub.hSub n 2)) ↑n) (HDiv.hDiv ↑(HPow.hPow d  …
  -/
  rw [cast_mul, cast_pow, mul_comm, ← div_div, pow_sub₀ _ _ hn, ← div_eq_mul_inv, cast_pow]
  /-
    n d : Nat
    hd : Ne d 0
    hn : LE.le 2 n
    ⊢ Ne (↑d) 0
  -/
  rwa [cast_ne_zero]
  /-
    🎉 no goals
  -/


theorem log_two_mul_two_le_sqrt_log_eight : log 2 * 2 ≤ √(log 8) := by
  /-
    ⊢ LE.le (HMul.hMul (Real.log 2) 2) (Real.log 8).sqrt
  -/
  have : (8 : ℝ) = 2 ^ ((3 : ℕ) : ℝ) := by rw [rpow_natCast]; norm_num
  /-
    this : Eq 8 (HPow.hPow 2 ↑3)
    ⊢ LE.le (HMul.hMul (Real.log 2) 2) (Real.log 8).sqrt
  -/
  rw [this, log_rpow zero_lt_two (3 : ℕ)]
  /-
    this : Eq 8 (HPow.hPow 2 ↑3)
    ⊢ LE.le (HMul.hMul (Real.log 2) 2) (HMul.hMul (↑3) (Real.log 2)).sqrt
  -/
  apply le_sqrt_of_sq_le
  /-
    case h
    this : Eq 8 (HPow.hPow 2 ↑3)
    ⊢ LE.le (HPow.hPow (HMul.hMul (Real.log 2) 2) 2) (HMul.hMul (↑3) (Real.log 2))
  -/
  rw [mul_pow, sq (log 2), mul_assoc, mul_comm]
  /-
    case h
    this : Eq 8 (HPow.hPow 2 ↑3)
    ⊢ LE.le (HMul.hMul (HMul.hMul (Real.log 2) (HPow.hPow 2 2)) (Real.log 2)) (HMu …
  -/
  refine mul_le_mul_of_nonneg_right ?_ (log_nonneg one_le_two)
  /-
    case h
    this : Eq 8 (HPow.hPow 2 ↑3)
    ⊢ LE.le (HMul.hMul (Real.log 2) (HPow.hPow 2 2)) ↑3
  -/
  rw [← le_div_iff₀]
  /-
    case h
    this : Eq 8 (HPow.hPow 2 ↑3)
    ⊢ LE.le (Real.log 2) (HDiv.hDiv (↑3) (HPow.hPow 2 2))
  -/
  on_goal 1 => apply log_two_lt_d9.le.trans
  /-
    case h
    this : Eq 8 (HPow.hPow 2 ↑3)
    ⊢ LE.le 0.6931471808 (HDiv.hDiv (↑3) (HPow.hPow 2 2))
  -/
  all_goals norm_num1
  /-
    🎉 no goals
  -/


theorem two_div_one_sub_two_div_e_le_eight : 2 / (1 - 2 / exp 1) ≤ 8 := by
  /-
    ⊢ LE.le (HDiv.hDiv 2 (HSub.hSub 1 (HDiv.hDiv 2 (Real.exp 1)))) 8
  -/
  rw [div_le_iff₀, mul_sub, mul_one, mul_div_assoc', le_sub_comm, div_le_iff₀ (exp_pos _)]
    /-
      ⊢ LE.le (HMul.hMul 8 2) (HMul.hMul (HSub.hSub 8 2) (Real.exp 1))
    -/
  · linarith [exp_one_gt_d9]
    /-
      🎉 no goals
    -/
  /-
    ⊢ LT.lt 0 (HSub.hSub 1 (HDiv.hDiv 2 (Real.exp 1)))
  -/
                               /-
                                 🎉 no goals
                               -/
  rw [sub_pos, div_lt_one] <;> exact exp_one_gt_d9.trans' (by norm_num)
                               /-
                                 🎉 no goals
                               -/


theorem le_sqrt_log (hN : 4096 ≤ N) : log (2 / (1 - 2 / exp 1)) * (69 / 50) ≤ √(log ↑N) := by
  have : (12 : ℕ) * log 2 ≤ log N := by
    rw [← log_rpow zero_lt_two, rpow_natCast]
    exact log_le_log (by positivity) (mod_cast hN)
  refine (mul_le_mul_of_nonneg_right (log_le_log ?_ two_div_one_sub_two_div_e_le_eight) <| by
    norm_num1).trans ?_
    /-
      case refine_1
      N : Nat
      hN : LE.le 4096 N
      this : LE.le (HMul.hMul (↑12) (Real.log 2)) (Real.log ↑N)
      ⊢ LT.lt 0 (HDiv.hDiv 2 (HSub.hSub 1 (HDiv.hDiv 2 (Real.exp 1))))
    -/
  · refine div_pos zero_lt_two ?_
    /-
      case refine_1
      N : Nat
      hN : LE.le 4096 N
      this : LE.le (HMul.hMul (↑12) (Real.log 2)) (Real.log ↑N)
      ⊢ LT.lt 0 (HSub.hSub 1 (HDiv.hDiv 2 (Real.exp 1)))
    -/
    rw [sub_pos, div_lt_one (exp_pos _)]
    /-
      case refine_1
      N : Nat
      hN : LE.le 4096 N
      this : LE.le (HMul.hMul (↑12) (Real.log 2)) (Real.log ↑N)
      ⊢ LT.lt 2 (Real.exp 1)
    -/
    exact exp_one_gt_d9.trans_le' (by norm_num1)
    /-
      🎉 no goals
    -/
  have l8 : log 8 = (3 : ℕ) * log 2 := by
    rw [← log_rpow zero_lt_two, rpow_natCast]
    norm_num
  /-
    case refine_2
    N : Nat
    hN : LE.le 4096 N
    this : LE.le (HMul.hMul (↑12) (Real.log 2)) (Real.log ↑N)
    l8 : Eq (Real.log 8) (HMul.hMul (↑3) (Real.log 2))
    ⊢ LE.le (HMul.hMul (Real.log 8) (69 / 50)) (Real.log ↑N).sqrt
  -/
  rw [l8]
  /-
    case refine_2
    N : Nat
    hN : LE.le 4096 N
    this : LE.le (HMul.hMul (↑12) (Real.log 2)) (Real.log ↑N)
    l8 : Eq (Real.log 8) (HMul.hMul (↑3) (Real.log 2))
    ⊢ LE.le (HMul.hMul (HMul.hMul (↑3) (Real.log 2)) (69 / 50)) (Real.log ↑N).sqrt
  -/
  apply le_sqrt_of_sq_le (le_trans _ this)
  /-
    N : Nat
    hN : LE.le 4096 N
    this : LE.le (HMul.hMul (↑12) (Real.log 2)) (Real.log ↑N)
    l8 : Eq (Real.log 8) (HMul.hMul (↑3) (Real.log 2))
    ⊢ LE.le (HPow.hPow (HMul.hMul (HMul.hMul (↑3) (Real.log 2)) (69 / 50)) 2) (HMu …
  -/
  rw [mul_right_comm, mul_pow, sq (log 2), ← mul_assoc]
  /-
    N : Nat
    hN : LE.le 4096 N
    this : LE.le (HMul.hMul (↑12) (Real.log 2)) (Real.log ↑N)
    l8 : Eq (Real.log 8) (HMul.hMul (↑3) (Real.log 2))
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (HMul.hMul (↑3) (69 / 50)) 2) (Real.l …
  -/
  apply mul_le_mul_of_nonneg_right _ (log_nonneg one_le_two)
  /-
    N : Nat
    hN : LE.le 4096 N
    this : LE.le (HMul.hMul (↑12) (Real.log 2)) (Real.log ↑N)
    l8 : Eq (Real.log 8) (HMul.hMul (↑3) (Real.log 2))
    ⊢ LE.le (HMul.hMul (HPow.hPow (HMul.hMul (↑3) (69 / 50)) 2) (Real.log 2)) ↑12
  -/
  rw [← le_div_iff₀']
    /-
      N : Nat
      hN : LE.le 4096 N
      this : LE.le (HMul.hMul (↑12) (Real.log 2)) (Real.log ↑N)
      l8 : Eq (Real.log 8) (HMul.hMul (↑3) (Real.log 2))
      ⊢ LE.le (Real.log 2) (HDiv.hDiv (↑12) (HPow.hPow (HMul.hMul (↑3) (69 / 50)) 2))
    -/
  · exact log_two_lt_d9.le.trans (by norm_num1)
    /-
      🎉 no goals
    -/
  /-
    N : Nat
    hN : LE.le 4096 N
    this : LE.le (HMul.hMul (↑12) (Real.log 2)) (Real.log ↑N)
    l8 : Eq (Real.log 8) (HMul.hMul (↑3) (Real.log 2))
    ⊢ LT.lt 0 (HPow.hPow (HMul.hMul (↑3) (69 / 50)) 2)
  -/
  exact sq_pos_of_ne_zero (by norm_num1)
  /-
    🎉 no goals
  -/


theorem exp_neg_two_mul_le {x : ℝ} (hx : 0 < x) : exp (-2 * x) < exp (2 - ⌈x⌉₊) / ⌈x⌉₊ := by
  /-
    x : Real
    hx : LT.lt 0 x
    ⊢ LT.lt (Real.exp (HMul.hMul (-2) x)) (HDiv.hDiv (Real.exp (HSub.hSub 2 ↑(Nat. …
  -/
  have h₁ := ceil_lt_add_one hx.le
  /-
    x : Real
    hx : LT.lt 0 x
    h₁ : LT.lt (↑(Nat.ceil x)) (HAdd.hAdd x 1)
    ⊢ LT.lt (Real.exp (HMul.hMul (-2) x)) (HDiv.hDiv (Real.exp (HSub.hSub 2 ↑(Nat. …
  -/
  have h₂ : 1 - x ≤ 2 - ⌈x⌉₊ := by linarith
  calc
    _ ≤ exp (1 - x) / (x + 1) := ?_
    _ ≤ exp (2 - ⌈x⌉₊) / (x + 1) := by gcongr
    _ < _ := by gcongr
  rw [le_div_iff₀  (add_pos hx zero_lt_one), ← le_div_iff₀' (exp_pos _), ← exp_sub, neg_mul,
    sub_neg_eq_add, two_mul, sub_add_add_cancel, add_comm _ x]
  /-
    x : Real
    hx : LT.lt 0 x
    h₁ : LT.lt (↑(Nat.ceil x)) (HAdd.hAdd x 1)
    h₂ : LE.le (HSub.hSub 1 x) (HSub.hSub 2 ↑(Nat.ceil x))
    ⊢ LE.le (HAdd.hAdd x 1) (Real.exp (HAdd.hAdd x 1))
  -/
  exact le_trans (le_add_of_nonneg_right zero_le_one) (add_one_le_exp _)
  /-
    🎉 no goals
  -/


theorem div_lt_floor {x : ℝ} (hx : 2 / (1 - 2 / exp 1) ≤ x) : x / exp 1 < (⌊x / 2⌋₊ : ℝ) := by
  /-
    x : Real
    hx : LE.le (HDiv.hDiv 2 (HSub.hSub 1 (HDiv.hDiv 2 (Real.exp 1)))) x
    ⊢ LT.lt (HDiv.hDiv x (Real.exp 1)) ↑(Nat.floor (HDiv.hDiv x 2))
  -/
  apply lt_of_le_of_lt _ (sub_one_lt_floor _)
  have : 0 < 1 - 2 / exp 1 := by
    rw [sub_pos, div_lt_one (exp_pos _)]
    exact lt_of_le_of_lt (by norm_num) exp_one_gt_d9
  rwa [le_sub_comm, div_eq_mul_one_div x, div_eq_mul_one_div x, ← mul_sub, div_sub', ←
    div_eq_mul_one_div, mul_div_assoc', one_le_div, ← div_le_iff₀ this]
    /-
      x : Real
      hx : LE.le (HDiv.hDiv 2 (HSub.hSub 1 (HDiv.hDiv 2 (Real.exp 1)))) x
      this : LT.lt 0 (HSub.hSub 1 (HDiv.hDiv 2 (Real.exp 1)))
      ⊢ LT.lt 0 2
    -/
  · exact zero_lt_two
    /-
      🎉 no goals
    -/
    /-
      case hc
      x : Real
      hx : LE.le (HDiv.hDiv 2 (HSub.hSub 1 (HDiv.hDiv 2 (Real.exp 1)))) x
      this : LT.lt 0 (HSub.hSub 1 (HDiv.hDiv 2 (Real.exp 1)))
      ⊢ Ne 2 0
    -/
  · exact two_ne_zero
    /-
      🎉 no goals
    -/


theorem ceil_lt_mul {x : ℝ} (hx : 50 / 19 ≤ x) : (⌈x⌉₊ : ℝ) < 1.38 * x := by
  /-
    x : Real
    hx : LE.le (50 / 19) x
    ⊢ LT.lt (↑(Nat.ceil x)) (HMul.hMul 1.38 x)
  -/
  refine (ceil_lt_add_one <| hx.trans' <| by norm_num).trans_le ?_
  /-
    x : Real
    hx : LE.le (50 / 19) x
    ⊢ LE.le (HAdd.hAdd x 1) (HMul.hMul 1.38 x)
  -/
  rw [← le_sub_iff_add_le', ← sub_one_mul]
  /-
    x : Real
    hx : LE.le (50 / 19) x
    ⊢ LE.le 1 (HMul.hMul (HSub.hSub 1.38 1) x)
  -/
  have : (1.38 : ℝ) = 69 / 50 := by norm_num
  rwa [this, show (69 / 50 - 1 : ℝ) = (50 / 19)⁻¹ by norm_num1, ←
    div_eq_inv_mul, one_le_div]
  /-
    x : Real
    hx : LE.le (50 / 19) x
    this : Eq 1.38 (69 / 50)
    ⊢ LT.lt 0 (50 / 19)
  -/
  norm_num1
  /-
    🎉 no goals
  -/


/-- The (almost) optimal value of `n` in `Behrend.bound_aux`. -/
noncomputable def nValue (N : ℕ) : ℕ :=
  ⌈√(log N)⌉₊


/-- The (almost) optimal value of `d` in `Behrend.bound_aux`. -/
noncomputable def dValue (N : ℕ) : ℕ := ⌊(N : ℝ) ^ (nValue N : ℝ)⁻¹ / 2⌋₊


theorem nValue_pos (hN : 2 ≤ N) : 0 < nValue N :=
  ceil_pos.2 <| Real.sqrt_pos.2 <| log_pos <| one_lt_cast.2 <| hN


theorem three_le_nValue (hN : 64 ≤ N) : 3 ≤ nValue N := by
  /-
    N : Nat
    hN : LE.le 64 N
    ⊢ LE.le 3 (Behrend.nValue N)
  -/
  rw [nValue, ← lt_iff_add_one_le, lt_ceil, cast_two]
  /-
    N : Nat
    hN : LE.le 64 N
    ⊢ LT.lt 2 (Real.log ↑N).sqrt
  -/
  apply lt_sqrt_of_sq_lt
  have : (2 : ℝ) ^ ((6 : ℕ) : ℝ) ≤ N := by
    rw [rpow_natCast]
    exact (cast_le.2 hN).trans' (by norm_num1)
  /-
    case h
    N : Nat
    hN : LE.le 64 N
    this : LE.le (HPow.hPow 2 ↑6) ↑N
    ⊢ LT.lt (HPow.hPow 2 2) (Real.log ↑N)
  -/
  apply lt_of_lt_of_le _ (log_le_log (rpow_pos_of_pos zero_lt_two _) this)
  /-
    N : Nat
    hN : LE.le 64 N
    this : LE.le (HPow.hPow 2 ↑6) ↑N
    ⊢ LT.lt (HPow.hPow 2 2) (Real.log (HPow.hPow 2 ↑6))
  -/
  rw [log_rpow zero_lt_two, ← div_lt_iff₀']
    /-
      N : Nat
      hN : LE.le 64 N
      this : LE.le (HPow.hPow 2 ↑6) ↑N
      ⊢ LT.lt (HDiv.hDiv (HPow.hPow 2 2) ↑6) (Real.log 2)
    -/
  · exact log_two_gt_d9.trans_le' (by norm_num1)
    /-
      🎉 no goals
    -/
    /-
      N : Nat
      hN : LE.le 64 N
      this : LE.le (HPow.hPow 2 ↑6) ↑N
      ⊢ LT.lt 0 ↑6
    -/
  · norm_num1
    /-
      🎉 no goals
    -/


theorem dValue_pos (hN₃ : 8 ≤ N) : 0 < dValue N := by
  /-
    N : Nat
    hN₃ : LE.le 8 N
    ⊢ LT.lt 0 (Behrend.dValue N)
  -/
  have hN₀ : 0 < (N : ℝ) := cast_pos.2 (succ_pos'.trans_le hN₃)
  rw [dValue, floor_pos, ← log_le_log_iff zero_lt_one, log_one, log_div _ two_ne_zero, log_rpow hN₀,
    inv_mul_eq_div, sub_nonneg, le_div_iff₀]
  · have : (nValue N : ℝ) ≤ 2 * √(log N) := by
      apply (ceil_lt_add_one <| sqrt_nonneg _).le.trans
      rw [two_mul, add_le_add_iff_left]
      apply le_sqrt_of_sq_le
      rw [one_pow, le_log_iff_exp_le hN₀]
      exact (exp_one_lt_d9.le.trans <| by norm_num).trans (cast_le.2 hN₃)
    /-
      N : Nat
      hN₃ : LE.le 8 N
      hN₀ : LT.lt 0 ↑N
      this : LE.le (↑(Behrend.nValue N)) (HMul.hMul 2 (Real.log ↑N).sqrt)
      ⊢ LE.le (HMul.hMul (Real.log 2) ↑(Behrend.nValue N)) (Real.log ↑N)
    -/
    apply (mul_le_mul_of_nonneg_left this <| log_nonneg one_le_two).trans _
    /-
      N : Nat
      hN₃ : LE.le 8 N
      hN₀ : LT.lt 0 ↑N
      this : LE.le (↑(Behrend.nValue N)) (HMul.hMul 2 (Real.log ↑N).sqrt)
      ⊢ LE.le (HMul.hMul (Real.log 2) (HMul.hMul 2 (Real.log ↑N).sqrt)) (Real.log ↑N)
    -/
    rw [← mul_assoc, ← le_div_iff₀ (Real.sqrt_pos.2 <| log_pos <| one_lt_cast.2 _), div_sqrt]
      /-
        N : Nat
        hN₃ : LE.le 8 N
        hN₀ : LT.lt 0 ↑N
        this : LE.le (↑(Behrend.nValue N)) (HMul.hMul 2 (Real.log ↑N).sqrt)
        ⊢ LE.le (HMul.hMul (Real.log 2) 2) (Real.log ↑N).sqrt
      -/
    · apply log_two_mul_two_le_sqrt_log_eight.trans
      /-
        N : Nat
        hN₃ : LE.le 8 N
        hN₀ : LT.lt 0 ↑N
        this : LE.le (↑(Behrend.nValue N)) (HMul.hMul 2 (Real.log ↑N).sqrt)
        ⊢ LE.le (Real.log 8).sqrt (Real.log ↑N).sqrt
      -/
      apply Real.sqrt_le_sqrt
      /-
        case h
        N : Nat
        hN₃ : LE.le 8 N
        hN₀ : LT.lt 0 ↑N
        this : LE.le (↑(Behrend.nValue N)) (HMul.hMul 2 (Real.log ↑N).sqrt)
        ⊢ LE.le (Real.log 8) (Real.log ↑N)
      -/
      exact log_le_log (by norm_num) (mod_cast hN₃)
      /-
        🎉 no goals
      -/
    /-
      N : Nat
      hN₃ : LE.le 8 N
      hN₀ : LT.lt 0 ↑N
      this : LE.le (↑(Behrend.nValue N)) (HMul.hMul 2 (Real.log ↑N).sqrt)
      ⊢ LT.lt 1 N
    -/
    exact hN₃.trans_lt' (by norm_num)
    /-
      🎉 no goals
    -/
    /-
      N : Nat
      hN₃ : LE.le 8 N
      hN₀ : LT.lt 0 ↑N
      ⊢ LT.lt 0 ↑(Behrend.nValue N)
    -/
  · exact cast_pos.2 (nValue_pos <| hN₃.trans' <| by norm_num)
    /-
      🎉 no goals
    -/
    /-
      N : Nat
      hN₃ : LE.le 8 N
      hN₀ : LT.lt 0 ↑N
      ⊢ Ne (HPow.hPow (↑N) (Inv.inv ↑(Behrend.nValue N))) 0
    -/
  · exact (rpow_pos_of_pos hN₀ _).ne'
    /-
      🎉 no goals
    -/
    /-
      N : Nat
      hN₃ : LE.le 8 N
      hN₀ : LT.lt 0 ↑N
      ⊢ LT.lt 0 (HDiv.hDiv (HPow.hPow (↑N) (Inv.inv ↑(Behrend.nValue N))) 2)
    -/
  · exact div_pos (rpow_pos_of_pos hN₀ _) zero_lt_two
    /-
      🎉 no goals
    -/


theorem le_N (hN : 2 ≤ N) : (2 * dValue N - 1) ^ nValue N ≤ N := by
  have : (2 * dValue N - 1) ^ nValue N ≤ (2 * dValue N) ^ nValue N :=
    Nat.pow_le_pow_left (Nat.sub_le _ _) _
  /-
    N : Nat
    hN : LE.le 2 N
    this : LE.le (HPow.hPow (HSub.hSub (HMul.hMul 2 (Behrend.dValue N)) 1) (Behren …
    ⊢ LE.le (HPow.hPow (HSub.hSub (HMul.hMul 2 (Behrend.dValue N)) 1) (Behrend.nVa …
  -/
  apply this.trans
  /-
    N : Nat
    hN : LE.le 2 N
    this : LE.le (HPow.hPow (HSub.hSub (HMul.hMul 2 (Behrend.dValue N)) 1) (Behren …
    ⊢ LE.le (HPow.hPow (HMul.hMul 2 (Behrend.dValue N)) (Behrend.nValue N)) N
  -/
  suffices ((2 * dValue N) ^ nValue N : ℝ) ≤ N from mod_cast this
  suffices i : (2 * dValue N : ℝ) ≤ (N : ℝ) ^ (nValue N : ℝ)⁻¹ by
    rw [← rpow_natCast]
    apply (rpow_le_rpow (mul_nonneg zero_le_two (cast_nonneg _)) i (cast_nonneg _)).trans
    rw [← rpow_mul (cast_nonneg _), inv_mul_cancel₀, rpow_one]
    rw [cast_ne_zero]
    apply (nValue_pos hN).ne'
  /-
    N : Nat
    hN : LE.le 2 N
    this : LE.le (HPow.hPow (HSub.hSub (HMul.hMul 2 (Behrend.dValue N)) 1) (Behren …
    ⊢ LE.le (HMul.hMul 2 ↑(Behrend.dValue N)) (HPow.hPow (↑N) (Inv.inv ↑(Behrend.n …
  -/
  rw [← le_div_iff₀']
    /-
      N : Nat
      hN : LE.le 2 N
      this : LE.le (HPow.hPow (HSub.hSub (HMul.hMul 2 (Behrend.dValue N)) 1) (Behren …
      ⊢ LE.le (↑(Behrend.dValue N)) (HDiv.hDiv (HPow.hPow (↑N) (Inv.inv ↑(Behrend.nV …
    -/
  · exact floor_le (div_nonneg (rpow_nonneg (cast_nonneg _) _) zero_le_two)
    /-
      🎉 no goals
    -/
  /-
    N : Nat
    hN : LE.le 2 N
    this : LE.le (HPow.hPow (HSub.hSub (HMul.hMul 2 (Behrend.dValue N)) 1) (Behren …
    ⊢ LT.lt 0 2
  -/
  apply zero_lt_two
  /-
    🎉 no goals
  -/


theorem bound (hN : 4096 ≤ N) : (N : ℝ) ^ (nValue N : ℝ)⁻¹ / exp 1 < dValue N := by
  /-
    N : Nat
    hN : LE.le 4096 N
    ⊢ LT.lt (HDiv.hDiv (HPow.hPow (↑N) (Inv.inv ↑(Behrend.nValue N))) (Real.exp 1) …
  -/
  apply div_lt_floor _
  /-
    N : Nat
    hN : LE.le 4096 N
    ⊢ LE.le (HDiv.hDiv 2 (HSub.hSub 1 (HDiv.hDiv 2 (Real.exp 1)))) (HPow.hPow (↑N) …
  -/
  rw [← log_le_log_iff, log_rpow, mul_comm, ← div_eq_mul_inv]
    /-
      N : Nat
      hN : LE.le 4096 N
      ⊢ LE.le (Real.log (HDiv.hDiv 2 (HSub.hSub 1 (HDiv.hDiv 2 (Real.exp 1))))) (HDi …
    -/
  · apply le_trans _ (div_le_div_of_nonneg_left _ _ (ceil_lt_mul _).le)
      /-
        N : Nat
        hN : LE.le 4096 N
        ⊢ LE.le (Real.log (HDiv.hDiv 2 (HSub.hSub 1 (HDiv.hDiv 2 (Real.exp 1))))) (HDi …
      -/
    · rw [mul_comm, ← div_div, div_sqrt, le_div_iff₀]
        /-
          N : Nat
          hN : LE.le 4096 N
          ⊢ LE.le (HMul.hMul (Real.log (HDiv.hDiv 2 (HSub.hSub 1 (HDiv.hDiv 2 (Real.exp  …
        -/
      · norm_num; exact le_sqrt_log hN
                  /-
                    🎉 no goals
                  -/
        /-
          N : Nat
          hN : LE.le 4096 N
          ⊢ LT.lt 0 1.38
        -/
      · norm_num1
        /-
          🎉 no goals
        -/
      /-
        N : Nat
        hN : LE.le 4096 N
        ⊢ LE.le 0 (Real.log ↑N)
      -/
    · apply log_nonneg
      /-
        case hx
        N : Nat
        hN : LE.le 4096 N
        ⊢ LE.le 1 ↑N
      -/
      rw [one_le_cast]
      /-
        case hx
        N : Nat
        hN : LE.le 4096 N
        ⊢ LE.le 1 N
      -/
      exact hN.trans' (by norm_num1)
      /-
        🎉 no goals
      -/
      /-
        N : Nat
        hN : LE.le 4096 N
        ⊢ LT.lt 0 ↑(Nat.ceil (Real.log ↑N).sqrt)
      -/
    · rw [cast_pos, lt_ceil, cast_zero, Real.sqrt_pos]
      /-
        N : Nat
        hN : LE.le 4096 N
        ⊢ LT.lt 0 (Real.log ↑N)
      -/
      refine log_pos ?_
      /-
        N : Nat
        hN : LE.le 4096 N
        ⊢ LT.lt 1 ↑N
      -/
      rw [one_lt_cast]
      /-
        N : Nat
        hN : LE.le 4096 N
        ⊢ LT.lt 1 N
      -/
      exact hN.trans_lt' (by norm_num1)
      /-
        🎉 no goals
      -/
    /-
      N : Nat
      hN : LE.le 4096 N
      ⊢ LE.le (50 / 19) (Real.log ↑N).sqrt
    -/
    apply le_sqrt_of_sq_le
    have : (12 : ℕ) * log 2 ≤ log N := by
      rw [← log_rpow zero_lt_two, rpow_natCast]
      exact log_le_log (by positivity) (mod_cast hN)
    /-
      case h
      N : Nat
      hN : LE.le 4096 N
      this : LE.le (HMul.hMul (↑12) (Real.log 2)) (Real.log ↑N)
      ⊢ LE.le (HPow.hPow (50 / 19) 2) (Real.log ↑N)
    -/
    refine le_trans ?_ this
    /-
      case h
      N : Nat
      hN : LE.le 4096 N
      this : LE.le (HMul.hMul (↑12) (Real.log 2)) (Real.log ↑N)
      ⊢ LE.le (HPow.hPow (50 / 19) 2) (HMul.hMul (↑12) (Real.log 2))
    -/
    rw [← div_le_iff₀']
      /-
        case h
        N : Nat
        hN : LE.le 4096 N
        this : LE.le (HMul.hMul (↑12) (Real.log 2)) (Real.log ↑N)
        ⊢ LE.le (HDiv.hDiv (HPow.hPow (50 / 19) 2) ↑12) (Real.log 2)
      -/
    · exact log_two_gt_d9.le.trans' (by norm_num1)
      /-
        🎉 no goals
      -/
      /-
        case h
        N : Nat
        hN : LE.le 4096 N
        this : LE.le (HMul.hMul (↑12) (Real.log 2)) (Real.log ↑N)
        ⊢ LT.lt 0 ↑12
      -/
    · norm_num1
      /-
        🎉 no goals
      -/
    /-
      case hx
      N : Nat
      hN : LE.le 4096 N
      ⊢ LT.lt 0 ↑N
    -/
  · rw [cast_pos]
    /-
      case hx
      N : Nat
      hN : LE.le 4096 N
      ⊢ LT.lt 0 N
    -/
    exact hN.trans_lt' (by norm_num1)
    /-
      🎉 no goals
    -/
    /-
      case h
      N : Nat
      hN : LE.le 4096 N
      ⊢ LT.lt 0 (HDiv.hDiv 2 (HSub.hSub 1 (HDiv.hDiv 2 (Real.exp 1))))
    -/
  · refine div_pos zero_lt_two ?_
    /-
      case h
      N : Nat
      hN : LE.le 4096 N
      ⊢ LT.lt 0 (HSub.hSub 1 (HDiv.hDiv 2 (Real.exp 1)))
    -/
    rw [sub_pos, div_lt_one (exp_pos _)]
    /-
      case h
      N : Nat
      hN : LE.le 4096 N
      ⊢ LT.lt 2 (Real.exp 1)
    -/
    exact lt_of_le_of_lt (by norm_num1) exp_one_gt_d9
    /-
      🎉 no goals
    -/
  /-
    case h₁
    N : Nat
    hN : LE.le 4096 N
    ⊢ LT.lt 0 (HPow.hPow (↑N) (Inv.inv ↑(Behrend.nValue N)))
  -/
  positivity
  /-
    🎉 no goals
  -/


theorem roth_lower_bound_explicit (hN : 4096 ≤ N) :
    (N : ℝ) * exp (-4 * √(log N)) < rothNumberNat N := by
  /-
    N : Nat
    hN : LE.le 4096 N
    ⊢ LT.lt (HMul.hMul (↑N) (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt))) ↑(roth …
  -/
  let n := nValue N
  /-
    N : Nat
    hN : LE.le 4096 N
    n : Nat := Behrend.nValue N
    ⊢ LT.lt (HMul.hMul (↑N) (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt))) ↑(roth …
  -/
  have hn : 0 < (n : ℝ) := cast_pos.2 (nValue_pos <| hN.trans' <| by norm_num1)
  /-
    N : Nat
    hN : LE.le 4096 N
    n : Nat := Behrend.nValue N
    hn : LT.lt 0 ↑n
    ⊢ LT.lt (HMul.hMul (↑N) (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt))) ↑(roth …
  -/
  have hd : 0 < dValue N := dValue_pos (hN.trans' <| by norm_num1)
  /-
    N : Nat
    hN : LE.le 4096 N
    n : Nat := Behrend.nValue N
    hn : LT.lt 0 ↑n
    hd : LT.lt 0 (Behrend.dValue N)
    ⊢ LT.lt (HMul.hMul (↑N) (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt))) ↑(roth …
  -/
  have hN₀ : 0 < (N : ℝ) := cast_pos.2 (hN.trans' <| by norm_num1)
  /-
    N : Nat
    hN : LE.le 4096 N
    n : Nat := Behrend.nValue N
    hn : LT.lt 0 ↑n
    hd : LT.lt 0 (Behrend.dValue N)
    hN₀ : LT.lt 0 ↑N
    ⊢ LT.lt (HMul.hMul (↑N) (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt))) ↑(roth …
  -/
  have hn₂ : 2 < n := three_le_nValue <| hN.trans' <| by norm_num1
  /-
    N : Nat
    hN : LE.le 4096 N
    n : Nat := Behrend.nValue N
    hn : LT.lt 0 ↑n
    hd : LT.lt 0 (Behrend.dValue N)
    hN₀ : LT.lt 0 ↑N
    hn₂ : LT.lt 2 n
    ⊢ LT.lt (HMul.hMul (↑N) (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt))) ↑(roth …
  -/
  have : (2 * dValue N - 1) ^ n ≤ N := le_N (hN.trans' <| by norm_num1)
  calc
    _ ≤ (N ^ (nValue N : ℝ)⁻¹ / rexp 1 : ℝ) ^ (n - 2) / n := ?_
    _ < _ := by gcongr; exacts [(tsub_pos_of_lt hn₂).ne', bound hN]
    _ ≤ rothNumberNat ((2 * dValue N - 1) ^ n) := bound_aux hd.ne' hn₂.le
    _ ≤ rothNumberNat N := mod_cast rothNumberNat.mono this
  rw [← rpow_natCast, div_rpow (rpow_nonneg hN₀.le _) (exp_pos _).le, ← rpow_mul hN₀.le,
    inv_mul_eq_div, cast_sub hn₂.le, cast_two, same_sub_div hn.ne', exp_one_rpow,
    div_div, rpow_sub hN₀, rpow_one, div_div, div_eq_mul_inv]
  /-
    N : Nat
    hN : LE.le 4096 N
    n : Nat := Behrend.nValue N
    hn : LT.lt 0 ↑n
    hd : LT.lt 0 (Behrend.dValue N)
    hN₀ : LT.lt 0 ↑N
    hn₂ : LT.lt 2 n
    this : LE.le (HPow.hPow (HSub.hSub (HMul.hMul 2 (Behrend.dValue N)) 1) n) N
    ⊢ LE.le (HMul.hMul (↑N) (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt))) (HMul. …
  -/
  refine mul_le_mul_of_nonneg_left ?_ (cast_nonneg _)
  /-
    N : Nat
    hN : LE.le 4096 N
    n : Nat := Behrend.nValue N
    hn : LT.lt 0 ↑n
    hd : LT.lt 0 (Behrend.dValue N)
    hN₀ : LT.lt 0 ↑N
    hn₂ : LT.lt 2 n
    this : LE.le (HPow.hPow (HSub.hSub (HMul.hMul 2 (Behrend.dValue N)) 1) n) N
    ⊢ LE.le (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt)) (Inv.inv (HMul.hMul (HP …
  -/
  rw [mul_inv, mul_inv, ← exp_neg, ← rpow_neg (cast_nonneg _), neg_sub, ← div_eq_mul_inv]
  have : exp (-4 * √(log N)) = exp (-2 * √(log N)) * exp (-2 * √(log N)) := by
    rw [← exp_add, ← add_mul]
    norm_num
  /-
    N : Nat
    hN : LE.le 4096 N
    n : Nat := Behrend.nValue N
    hn : LT.lt 0 ↑n
    hd : LT.lt 0 (Behrend.dValue N)
    hN₀ : LT.lt 0 ↑N
    hn₂ : LT.lt 2 n
    this✝ : LE.le (HPow.hPow (HSub.hSub (HMul.hMul 2 (Behrend.dValue N)) 1) n) N
    this : Eq (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt)) (HMul.hMul (Real.exp  …
    ⊢ LE.le (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt)) (HMul.hMul (HPow.hPow ( …
  -/
  rw [this]
  refine mul_le_mul ?_ (exp_neg_two_mul_le <| Real.sqrt_pos.2 <| log_pos ?_).le (exp_pos _).le <|
      rpow_nonneg (cast_nonneg _) _
  · rw [← le_log_iff_exp_le (rpow_pos_of_pos hN₀ _), log_rpow hN₀, ← le_div_iff₀, mul_div_assoc,
      div_sqrt, neg_mul, neg_le_neg_iff, div_mul_eq_mul_div, div_le_iff₀ hn]
      /-
        case refine_1
        N : Nat
        hN : LE.le 4096 N
        n : Nat := Behrend.nValue N
        hn : LT.lt 0 ↑n
        hd : LT.lt 0 (Behrend.dValue N)
        hN₀ : LT.lt 0 ↑N
        hn₂ : LT.lt 2 n
        this✝ : LE.le (HPow.hPow (HSub.hSub (HMul.hMul 2 (Behrend.dValue N)) 1) n) N
        this : Eq (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt)) (HMul.hMul (Real.exp  …
        ⊢ LE.le (HMul.hMul 2 (Real.log ↑N).sqrt) (HMul.hMul 2 ↑n)
      -/
    · exact mul_le_mul_of_nonneg_left (le_ceil _) zero_le_two
      /-
        🎉 no goals
      -/
    /-
      case refine_1
      N : Nat
      hN : LE.le 4096 N
      n : Nat := Behrend.nValue N
      hn : LT.lt 0 ↑n
      hd : LT.lt 0 (Behrend.dValue N)
      hN₀ : LT.lt 0 ↑N
      hn₂ : LT.lt 2 n
      this✝ : LE.le (HPow.hPow (HSub.hSub (HMul.hMul 2 (Behrend.dValue N)) 1) n) N
      this : Eq (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt)) (HMul.hMul (Real.exp  …
      ⊢ LT.lt 0 (Real.log ↑N).sqrt
    -/
    refine Real.sqrt_pos.2 (log_pos ?_)
    /-
      case refine_1
      N : Nat
      hN : LE.le 4096 N
      n : Nat := Behrend.nValue N
      hn : LT.lt 0 ↑n
      hd : LT.lt 0 (Behrend.dValue N)
      hN₀ : LT.lt 0 ↑N
      hn₂ : LT.lt 2 n
      this✝ : LE.le (HPow.hPow (HSub.hSub (HMul.hMul 2 (Behrend.dValue N)) 1) n) N
      this : Eq (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt)) (HMul.hMul (Real.exp  …
      ⊢ LT.lt 1 ↑N
    -/
    rw [one_lt_cast]
    /-
      case refine_1
      N : Nat
      hN : LE.le 4096 N
      n : Nat := Behrend.nValue N
      hn : LT.lt 0 ↑n
      hd : LT.lt 0 (Behrend.dValue N)
      hN₀ : LT.lt 0 ↑N
      hn₂ : LT.lt 2 n
      this✝ : LE.le (HPow.hPow (HSub.hSub (HMul.hMul 2 (Behrend.dValue N)) 1) n) N
      this : Eq (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt)) (HMul.hMul (Real.exp  …
      ⊢ LT.lt 1 N
    -/
    exact hN.trans_lt' (by norm_num1)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      N : Nat
      hN : LE.le 4096 N
      n : Nat := Behrend.nValue N
      hn : LT.lt 0 ↑n
      hd : LT.lt 0 (Behrend.dValue N)
      hN₀ : LT.lt 0 ↑N
      hn₂ : LT.lt 2 n
      this✝ : LE.le (HPow.hPow (HSub.hSub (HMul.hMul 2 (Behrend.dValue N)) 1) n) N
      this : Eq (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt)) (HMul.hMul (Real.exp  …
      ⊢ LT.lt 1 ↑N
    -/
  · rw [one_lt_cast]
    /-
      case refine_2
      N : Nat
      hN : LE.le 4096 N
      n : Nat := Behrend.nValue N
      hn : LT.lt 0 ↑n
      hd : LT.lt 0 (Behrend.dValue N)
      hN₀ : LT.lt 0 ↑N
      hn₂ : LT.lt 2 n
      this✝ : LE.le (HPow.hPow (HSub.hSub (HMul.hMul 2 (Behrend.dValue N)) 1) n) N
      this : Eq (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt)) (HMul.hMul (Real.exp  …
      ⊢ LT.lt 1 N
    -/
    exact hN.trans_lt' (by norm_num1)
    /-
      🎉 no goals
    -/


theorem exp_four_lt : exp 4 < 64 := by
  rw [show (64 : ℝ) = 2 ^ ((6 : ℕ) : ℝ) by rw [rpow_natCast]; norm_num1,
    ← lt_log_iff_exp_lt (rpow_pos_of_pos zero_lt_two _), log_rpow zero_lt_two, ← div_lt_iff₀']
    /-
      ⊢ LT.lt (HDiv.hDiv 4 ↑6) (Real.log 2)
    -/
  · exact log_two_gt_d9.trans_le' (by norm_num1)
    /-
      🎉 no goals
    -/
    /-
      ⊢ LT.lt 0 ↑6
    -/
  · norm_num
    /-
      🎉 no goals
    -/


theorem four_zero_nine_six_lt_exp_sixteen : 4096 < exp 16 := by
  rw [← log_lt_iff_lt_exp (show (0 : ℝ) < 4096 by norm_num), show (4096 : ℝ) = 2 ^ 12 by norm_cast,
    ← rpow_natCast, log_rpow zero_lt_two, cast_ofNat]
  /-
    ⊢ LT.lt (HMul.hMul 12 (Real.log 2)) 16
  -/
  linarith [log_two_lt_d9]
  /-
    🎉 no goals
  -/


theorem lower_bound_le_one' (hN : 2 ≤ N) (hN' : N ≤ 4096) :
    (N : ℝ) * exp (-4 * √(log N)) ≤ 1 := by
  rw [← log_le_log_iff (mul_pos (cast_pos.2 (zero_lt_two.trans_le hN)) (exp_pos _)) zero_lt_one,
    log_one, log_mul (cast_pos.2 (zero_lt_two.trans_le hN)).ne' (exp_pos _).ne', log_exp, neg_mul, ←
    sub_eq_add_neg, sub_nonpos, ←
    div_le_iff₀ (Real.sqrt_pos.2 <| log_pos <| one_lt_cast.2 <| one_lt_two.trans_le hN), div_sqrt,
    sqrt_le_left zero_le_four, log_le_iff_le_exp (cast_pos.2 (zero_lt_two.trans_le hN))]
  /-
    N : Nat
    hN : LE.le 2 N
    hN' : LE.le N 4096
    ⊢ LE.le (↑N) (Real.exp (HPow.hPow 4 2))
  -/
  norm_num1
  /-
    N : Nat
    hN : LE.le 2 N
    hN' : LE.le N 4096
    ⊢ LE.le (↑N) (Real.exp 16)
  -/
  apply le_trans _ four_zero_nine_six_lt_exp_sixteen.le
  /-
    N : Nat
    hN : LE.le 2 N
    hN' : LE.le N 4096
    ⊢ LE.le (↑N) 4096
  -/
  exact mod_cast hN'
  /-
    🎉 no goals
  -/


theorem lower_bound_le_one (hN : 1 ≤ N) (hN' : N ≤ 4096) :
    (N : ℝ) * exp (-4 * √(log N)) ≤ 1 := by
  /-
    N : Nat
    hN : LE.le 1 N
    hN' : LE.le N 4096
    ⊢ LE.le (HMul.hMul (↑N) (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt))) 1
  -/
  obtain rfl | hN := hN.eq_or_lt
    /-
      case inl
      hN : LE.le 1 1
      hN' : LE.le 1 4096
      ⊢ LE.le (HMul.hMul (↑1) (Real.exp (HMul.hMul (-4) (Real.log ↑1).sqrt))) 1
    -/
  · norm_num
    /-
      🎉 no goals
    -/
    /-
      case inr
      N : Nat
      hN✝ : LE.le 1 N
      hN' : LE.le N 4096
      hN : LT.lt 1 N
      ⊢ LE.le (HMul.hMul (↑N) (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt))) 1
    -/
  · exact lower_bound_le_one' hN hN'
    /-
      🎉 no goals
    -/


theorem roth_lower_bound : (N : ℝ) * exp (-4 * √(log N)) ≤ rothNumberNat N := by
  /-
    N : Nat
    ⊢ LE.le (HMul.hMul (↑N) (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt))) ↑(roth …
  -/
  obtain rfl | hN := Nat.eq_zero_or_pos N
    /-
      case inl
      ⊢ LE.le (HMul.hMul (↑0) (Real.exp (HMul.hMul (-4) (Real.log ↑0).sqrt))) ↑(roth …
    -/
  · norm_num
    /-
      🎉 no goals
    -/
  /-
    case inr
    N : Nat
    hN : GT.gt N 0
    ⊢ LE.le (HMul.hMul (↑N) (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt))) ↑(roth …
  -/
  obtain h₁ | h₁ := le_or_lt 4096 N
    /-
      case inr.inl
      N : Nat
      hN : GT.gt N 0
      h₁ : LE.le 4096 N
      ⊢ LE.le (HMul.hMul (↑N) (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt))) ↑(roth …
    -/
  · exact (roth_lower_bound_explicit h₁).le
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      N : Nat
      hN : GT.gt N 0
      h₁ : LT.lt N 4096
      ⊢ LE.le (HMul.hMul (↑N) (Real.exp (HMul.hMul (-4) (Real.log ↑N).sqrt))) ↑(roth …
    -/
  · apply (lower_bound_le_one hN h₁.le).trans
    /-
      case inr.inr
      N : Nat
      hN : GT.gt N 0
      h₁ : LT.lt N 4096
      ⊢ LE.le 1 ↑(rothNumberNat N)
    -/
    simpa using rothNumberNat.monotone hN
    /-
      🎉 no goals
    -/


