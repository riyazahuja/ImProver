/-- In a tetrahedron with vertices `x`, `y`, `p`, `q`, any segment `[u, v]` joining the opposite
edges `[x, p]` and `[y, q]` passes through any triangle of vertices `p`, `q`, `z` where
`z ∈ [x, y]`. -/
theorem not_disjoint_segment_convexHull_triple {p q u v x y z : E} (hz : z ∈ segment 𝕜 x y)
    (hu : u ∈ segment 𝕜 x p) (hv : v ∈ segment 𝕜 y q) :
    ¬Disjoint (segment 𝕜 u v) (convexHull 𝕜 {p, q, z}) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p q u v x y z : E
    hz : Membership.mem (segment 𝕜 x y) z
    hu : Membership.mem (segment 𝕜 x p) u
    hv : Membership.mem (segment 𝕜 y q) v
    ⊢ Not (Disjoint (segment 𝕜 u v) ((convexHull 𝕜) (Insert.insert p (Insert.inser …
  -/
  rw [not_disjoint_iff]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p q u v x y z : E
    hz : Membership.mem (segment 𝕜 x y) z
    hu : Membership.mem (segment 𝕜 x p) u
    hv : Membership.mem (segment 𝕜 y q) v
    ⊢ Exists fun x => And (Membership.mem (segment 𝕜 u v) x) (Membership.mem ((con …
  -/
  obtain ⟨az, bz, haz, hbz, habz, rfl⟩ := hz
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p q u v x y : E
    hu : Membership.mem (segment 𝕜 x p) u
    hv : Membership.mem (segment 𝕜 y q) v
    az bz : 𝕜
    haz : LE.le 0 az
    hbz : LE.le 0 bz
    habz : Eq (HAdd.hAdd az bz) 1
    ⊢ Exists fun x_1 => And (Membership.mem (segment 𝕜 u v) x_1) (Membership.mem ( …
  -/
  obtain rfl | haz' := haz.eq_or_lt
    /-
      case intro.intro.intro.intro.intro.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p q u v x y : E
      hu : Membership.mem (segment 𝕜 x p) u
      hv : Membership.mem (segment 𝕜 y q) v
      bz : 𝕜
      hbz : LE.le 0 bz
      haz : LE.le 0 0
      habz : Eq (HAdd.hAdd 0 bz) 1
      ⊢ Exists fun x_1 => And (Membership.mem (segment 𝕜 u v) x_1) (Membership.mem ( …
    -/
  · rw [zero_add] at habz
    /-
      case intro.intro.intro.intro.intro.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p q u v x y : E
      hu : Membership.mem (segment 𝕜 x p) u
      hv : Membership.mem (segment 𝕜 y q) v
      bz : 𝕜
      hbz : LE.le 0 bz
      haz : LE.le 0 0
      habz : Eq bz 1
      ⊢ Exists fun x_1 => And (Membership.mem (segment 𝕜 u v) x_1) (Membership.mem ( …
    -/
    rw [zero_smul, zero_add, habz, one_smul]
    /-
      case intro.intro.intro.intro.intro.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p q u v x y : E
      hu : Membership.mem (segment 𝕜 x p) u
      hv : Membership.mem (segment 𝕜 y q) v
      bz : 𝕜
      hbz : LE.le 0 bz
      haz : LE.le 0 0
      habz : Eq bz 1
      ⊢ Exists fun x => And (Membership.mem (segment 𝕜 u v) x) (Membership.mem ((con …
    -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
    refine ⟨v, by apply right_mem_segment, segment_subset_convexHull ?_ ?_ hv⟩ <;> simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
  /-
    case intro.intro.intro.intro.intro.inr
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p q u v x y : E
    hu : Membership.mem (segment 𝕜 x p) u
    hv : Membership.mem (segment 𝕜 y q) v
    az bz : 𝕜
    haz : LE.le 0 az
    hbz : LE.le 0 bz
    habz : Eq (HAdd.hAdd az bz) 1
    haz' : LT.lt 0 az
    ⊢ Exists fun x_1 => And (Membership.mem (segment 𝕜 u v) x_1) (Membership.mem ( …
  -/
  obtain ⟨av, bv, hav, hbv, habv, rfl⟩ := hv
  /-
    case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p q u x y : E
    hu : Membership.mem (segment 𝕜 x p) u
    az bz : 𝕜
    haz : LE.le 0 az
    hbz : LE.le 0 bz
    habz : Eq (HAdd.hAdd az bz) 1
    haz' : LT.lt 0 az
    av bv : 𝕜
    hav : LE.le 0 av
    hbv : LE.le 0 bv
    habv : Eq (HAdd.hAdd av bv) 1
    ⊢ Exists fun x_1 => And (Membership.mem (segment 𝕜 u (HAdd.hAdd (HSMul.hSMul a …
  -/
  obtain rfl | hav' := hav.eq_or_lt
    /-
      case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro.intro.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p q u x y : E
      hu : Membership.mem (segment 𝕜 x p) u
      az bz : 𝕜
      haz : LE.le 0 az
      hbz : LE.le 0 bz
      habz : Eq (HAdd.hAdd az bz) 1
      haz' : LT.lt 0 az
      bv : 𝕜
      hbv : LE.le 0 bv
      hav : LE.le 0 0
      habv : Eq (HAdd.hAdd 0 bv) 1
      ⊢ Exists fun x_1 => And (Membership.mem (segment 𝕜 u (HAdd.hAdd (HSMul.hSMul 0 …
    -/
  · rw [zero_add] at habv
    /-
      case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro.intro.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p q u x y : E
      hu : Membership.mem (segment 𝕜 x p) u
      az bz : 𝕜
      haz : LE.le 0 az
      hbz : LE.le 0 bz
      habz : Eq (HAdd.hAdd az bz) 1
      haz' : LT.lt 0 az
      bv : 𝕜
      hbv : LE.le 0 bv
      hav : LE.le 0 0
      habv : Eq bv 1
      ⊢ Exists fun x_1 => And (Membership.mem (segment 𝕜 u (HAdd.hAdd (HSMul.hSMul 0 …
    -/
    rw [zero_smul, zero_add, habv, one_smul]
    /-
      case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro.intro.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p q u x y : E
      hu : Membership.mem (segment 𝕜 x p) u
      az bz : 𝕜
      haz : LE.le 0 az
      hbz : LE.le 0 bz
      habz : Eq (HAdd.hAdd az bz) 1
      haz' : LT.lt 0 az
      bv : 𝕜
      hbv : LE.le 0 bv
      hav : LE.le 0 0
      habv : Eq bv 1
      ⊢ Exists fun x_1 => And (Membership.mem (segment 𝕜 u q) x_1) (Membership.mem ( …
    -/
    exact ⟨q, right_mem_segment _ _ _, subset_convexHull _ _ <| by simp⟩
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro.intro.inr
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p q u x y : E
    hu : Membership.mem (segment 𝕜 x p) u
    az bz : 𝕜
    haz : LE.le 0 az
    hbz : LE.le 0 bz
    habz : Eq (HAdd.hAdd az bz) 1
    haz' : LT.lt 0 az
    av bv : 𝕜
    hav : LE.le 0 av
    hbv : LE.le 0 bv
    habv : Eq (HAdd.hAdd av bv) 1
    hav' : LT.lt 0 av
    ⊢ Exists fun x_1 => And (Membership.mem (segment 𝕜 u (HAdd.hAdd (HSMul.hSMul a …
  -/
  obtain ⟨au, bu, hau, hbu, habu, rfl⟩ := hu
  /-
    case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro.intro.inr.intro …
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p q x y : E
    az bz : 𝕜
    haz : LE.le 0 az
    hbz : LE.le 0 bz
    habz : Eq (HAdd.hAdd az bz) 1
    haz' : LT.lt 0 az
    av bv : 𝕜
    hav : LE.le 0 av
    hbv : LE.le 0 bv
    habv : Eq (HAdd.hAdd av bv) 1
    hav' : LT.lt 0 av
    au bu : 𝕜
    hau : LE.le 0 au
    hbu : LE.le 0 bu
    habu : Eq (HAdd.hAdd au bu) 1
    ⊢ Exists fun x_1 => And (Membership.mem (segment 𝕜 (HAdd.hAdd (HSMul.hSMul au  …
  -/
  have hab : 0 < az * av + bz * au := by positivity
  refine ⟨(az * av / (az * av + bz * au)) • (au • x + bu • p) +
    (bz * au / (az * av + bz * au)) • (av • y + bv • q), ⟨_, _, ?_, ?_, ?_, rfl⟩, ?_⟩
    /-
      case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro.intro.inr.intro …
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p q x y : E
      az bz : 𝕜
      haz : LE.le 0 az
      hbz : LE.le 0 bz
      habz : Eq (HAdd.hAdd az bz) 1
      haz' : LT.lt 0 az
      av bv : 𝕜
      hav : LE.le 0 av
      hbv : LE.le 0 bv
      habv : Eq (HAdd.hAdd av bv) 1
      hav' : LT.lt 0 av
      au bu : 𝕜
      hau : LE.le 0 au
      hbu : LE.le 0 bu
      habu : Eq (HAdd.hAdd au bu) 1
      hab : LT.lt 0 (HAdd.hAdd (HMul.hMul az av) (HMul.hMul bz au))
      ⊢ LE.le 0 (HDiv.hDiv (HMul.hMul az av) (HAdd.hAdd (HMul.hMul az av) (HMul.hMul …
    -/
  · positivity
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro.intro.inr.intro …
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p q x y : E
      az bz : 𝕜
      haz : LE.le 0 az
      hbz : LE.le 0 bz
      habz : Eq (HAdd.hAdd az bz) 1
      haz' : LT.lt 0 az
      av bv : 𝕜
      hav : LE.le 0 av
      hbv : LE.le 0 bv
      habv : Eq (HAdd.hAdd av bv) 1
      hav' : LT.lt 0 av
      au bu : 𝕜
      hau : LE.le 0 au
      hbu : LE.le 0 bu
      habu : Eq (HAdd.hAdd au bu) 1
      hab : LT.lt 0 (HAdd.hAdd (HMul.hMul az av) (HMul.hMul bz au))
      ⊢ LE.le 0 (HDiv.hDiv (HMul.hMul bz au) (HAdd.hAdd (HMul.hMul az av) (HMul.hMul …
    -/
  · positivity
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro.intro.inr.intro …
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p q x y : E
      az bz : 𝕜
      haz : LE.le 0 az
      hbz : LE.le 0 bz
      habz : Eq (HAdd.hAdd az bz) 1
      haz' : LT.lt 0 az
      av bv : 𝕜
      hav : LE.le 0 av
      hbv : LE.le 0 bv
      habv : Eq (HAdd.hAdd av bv) 1
      hav' : LT.lt 0 av
      au bu : 𝕜
      hau : LE.le 0 au
      hbu : LE.le 0 bu
      habu : Eq (HAdd.hAdd au bu) 1
      hab : LT.lt 0 (HAdd.hAdd (HMul.hMul az av) (HMul.hMul bz au))
      ⊢ Eq (HAdd.hAdd (HDiv.hDiv (HMul.hMul az av) (HAdd.hAdd (HMul.hMul az av) (HMu …
    -/
  · rw [← add_div, div_self]; positivity
                              /-
                                🎉 no goals
                              -/
  classical
    let w : Fin 3 → 𝕜 := ![az * av * bu, bz * au * bv, au * av]
    let z : Fin 3 → E := ![p, q, az • x + bz • y]
    have hw₀ : ∀ i, 0 ≤ w i := by
      rintro i
      fin_cases i
      · exact mul_nonneg (mul_nonneg haz hav) hbu
      · exact mul_nonneg (mul_nonneg hbz hau) hbv
      · exact mul_nonneg hau hav
    have hw : ∑ i, w i = az * av + bz * au := by
      trans az * av * bu + (bz * au * bv + au * av)
      · simp [w, Fin.sum_univ_succ, Fin.sum_univ_zero]
      linear_combination (au * bv - 1 * au) * habz + (-(1 * az * au) + au) * habv + az * av * habu
    have hz : ∀ i, z i ∈ ({p, q, az • x + bz • y} : Set E) := fun i => by fin_cases i <;> simp [z]
    convert (Finset.centerMass_mem_convexHull (Finset.univ : Finset (Fin 3)) (fun i _ => hw₀ i)
        (by rwa [hw]) fun i _ => hz i : Finset.univ.centerMass w z ∈ _)
    rw [Finset.centerMass, hw]
    trans (az * av + bz * au)⁻¹ •
      ((az * av * bu) • p + ((bz * au * bv) • q + (au * av) • (az • x + bz • y)))
    · module
    congr 3
    simp [w, z]


/-- **Stone's Separation Theorem** -/
theorem exists_convex_convex_compl_subset (hs : Convex 𝕜 s) (ht : Convex 𝕜 t) (hst : Disjoint s t) :
    ∃ C : Set E, Convex 𝕜 C ∧ Convex 𝕜 Cᶜ ∧ s ⊆ C ∧ t ⊆ Cᶜ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    hs : Convex 𝕜 s
    ht : Convex 𝕜 t
    hst : Disjoint s t
    ⊢ Exists fun C => And (Convex 𝕜 C) (And (Convex 𝕜 (HasCompl.compl C)) (And (Ha …
  -/
  let S : Set (Set E) := { C | Convex 𝕜 C ∧ Disjoint C t }
  obtain ⟨C, hsC, hmax⟩ :=
    zorn_subset_nonempty S
      (fun c hcS hc ⟨_, _⟩ =>
        ⟨⋃₀ c,
          ⟨hc.directedOn.convex_sUnion fun s hs => (hcS hs).1,
            disjoint_sUnion_left.2 fun c hc => (hcS hc).2⟩,
          fun s => subset_sUnion_of_mem⟩)
      s ⟨hs, hst⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    hs : Convex 𝕜 s
    ht : Convex 𝕜 t
    hst : Disjoint s t
    S : Set (Set E) := setOf fun C => And (Convex 𝕜 C) (Disjoint C t)
    C : Set E
    hsC : HasSubset.Subset s C
    hmax : Maximal (fun x => Membership.mem S x) C
    ⊢ Exists fun C => And (Convex 𝕜 C) (And (Convex 𝕜 (HasCompl.compl C)) (And (Ha …
  -/
  obtain hC : _ ∧ _ := hmax.prop
  refine
    ⟨C, hC.1, convex_iff_segment_subset.2 fun x hx y hy z hz hzC => ?_, hsC, hC.2.subset_compl_left⟩
  suffices h : ∀ c ∈ Cᶜ, ∃ a ∈ C, (segment 𝕜 c a ∩ t).Nonempty by
    obtain ⟨p, hp, u, hu, hut⟩ := h x hx
    obtain ⟨q, hq, v, hv, hvt⟩ := h y hy
    refine
      not_disjoint_segment_convexHull_triple hz hu hv
        (hC.2.symm.mono (ht.segment_subset hut hvt) <| convexHull_min ?_ hC.1)
    simp [insert_subset_iff, hp, hq, singleton_subset_iff.2 hzC]
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    hs : Convex 𝕜 s
    ht : Convex 𝕜 t
    hst : Disjoint s t
    S : Set (Set E) := setOf fun C => And (Convex 𝕜 C) (Disjoint C t)
    C : Set E
    hsC : HasSubset.Subset s C
    hmax : Maximal (fun x => Membership.mem S x) C
    hC : And (Convex 𝕜 C) (Disjoint C t)
    x : E
    hx : Membership.mem (HasCompl.compl C) x
    y : E
    hy : Membership.mem (HasCompl.compl C) y
    z : E
    hz : Membership.mem (segment 𝕜 x y) z
    hzC : Membership.mem C z
    ⊢ ∀ (c : E), Membership.mem (HasCompl.compl C) c → Exists fun a => And (Member …
  -/
  rintro c hc
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    hs : Convex 𝕜 s
    ht : Convex 𝕜 t
    hst : Disjoint s t
    S : Set (Set E) := setOf fun C => And (Convex 𝕜 C) (Disjoint C t)
    C : Set E
    hsC : HasSubset.Subset s C
    hmax : Maximal (fun x => Membership.mem S x) C
    hC : And (Convex 𝕜 C) (Disjoint C t)
    x : E
    hx : Membership.mem (HasCompl.compl C) x
    y : E
    hy : Membership.mem (HasCompl.compl C) y
    z : E
    hz : Membership.mem (segment 𝕜 x y) z
    hzC : Membership.mem C z
    c : E
    hc : Membership.mem (HasCompl.compl C) c
    ⊢ Exists fun a => And (Membership.mem C a) (Inter.inter (segment 𝕜 c a) t).Non …
  -/
  by_contra! h
  suffices h : Disjoint (convexHull 𝕜 (insert c C)) t by
    rw [hmax.eq_of_subset ⟨convex_convexHull _ _, h⟩ <|
      (subset_insert ..).trans <| subset_convexHull ..] at hc
    exact hc (subset_convexHull _ _ <| mem_insert _ _)
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    hs : Convex 𝕜 s
    ht : Convex 𝕜 t
    hst : Disjoint s t
    S : Set (Set E) := setOf fun C => And (Convex 𝕜 C) (Disjoint C t)
    C : Set E
    hsC : HasSubset.Subset s C
    hmax : Maximal (fun x => Membership.mem S x) C
    hC : And (Convex 𝕜 C) (Disjoint C t)
    x : E
    hx : Membership.mem (HasCompl.compl C) x
    y : E
    hy : Membership.mem (HasCompl.compl C) y
    z : E
    hz : Membership.mem (segment 𝕜 x y) z
    hzC : Membership.mem C z
    c : E
    hc : Membership.mem (HasCompl.compl C) c
    h : ∀ (a : E), Membership.mem C a → Eq (Inter.inter (segment 𝕜 c a) t) EmptyCo …
    ⊢ Disjoint ((convexHull 𝕜) (Insert.insert c C)) t
  -/
  rw [convexHull_insert ⟨z, hzC⟩, convexJoin_singleton_left]
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    hs : Convex 𝕜 s
    ht : Convex 𝕜 t
    hst : Disjoint s t
    S : Set (Set E) := setOf fun C => And (Convex 𝕜 C) (Disjoint C t)
    C : Set E
    hsC : HasSubset.Subset s C
    hmax : Maximal (fun x => Membership.mem S x) C
    hC : And (Convex 𝕜 C) (Disjoint C t)
    x : E
    hx : Membership.mem (HasCompl.compl C) x
    y : E
    hy : Membership.mem (HasCompl.compl C) y
    z : E
    hz : Membership.mem (segment 𝕜 x y) z
    hzC : Membership.mem C z
    c : E
    hc : Membership.mem (HasCompl.compl C) c
    h : ∀ (a : E), Membership.mem C a → Eq (Inter.inter (segment 𝕜 c a) t) EmptyCo …
    ⊢ Disjoint (Set.iUnion fun y => Set.iUnion fun h => segment 𝕜 c y) t
  -/
  refine disjoint_iUnion₂_left.2 fun a ha => disjoint_iff_inter_eq_empty.2 (h a ?_)
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    hs : Convex 𝕜 s
    ht : Convex 𝕜 t
    hst : Disjoint s t
    S : Set (Set E) := setOf fun C => And (Convex 𝕜 C) (Disjoint C t)
    C : Set E
    hsC : HasSubset.Subset s C
    hmax : Maximal (fun x => Membership.mem S x) C
    hC : And (Convex 𝕜 C) (Disjoint C t)
    x : E
    hx : Membership.mem (HasCompl.compl C) x
    y : E
    hy : Membership.mem (HasCompl.compl C) y
    z : E
    hz : Membership.mem (segment 𝕜 x y) z
    hzC : Membership.mem C z
    c : E
    hc : Membership.mem (HasCompl.compl C) c
    h : ∀ (a : E), Membership.mem C a → Eq (Inter.inter (segment 𝕜 c a) t) EmptyCo …
    a : E
    ha : Membership.mem ((convexHull 𝕜) C) a
    ⊢ Membership.mem C a
  -/
  rwa [← hC.1.convexHull_eq]
  /-
    🎉 no goals
  -/

