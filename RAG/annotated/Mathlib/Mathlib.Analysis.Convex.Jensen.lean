/-- Convex **Jensen's inequality**, `Finset.centerMass` version. -/
theorem ConvexOn.map_centerMass_le (hf : ConvexOn 𝕜 s f) (h₀ : ∀ i ∈ t, 0 ≤ w i)
    (h₁ : 0 < ∑ i ∈ t, w i) (hmem : ∀ i ∈ t, p i ∈ s) :
    f (t.centerMass w p) ≤ t.centerMass w (f ∘ p) := by
  have hmem' : ∀ i ∈ t, (p i, (f ∘ p) i) ∈ { p : E × β | p.1 ∈ s ∧ f p.1 ≤ p.2 } := fun i hi =>
    ⟨hmem i hi, le_rfl⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    ι : Type u_5
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : OrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    t : Finset ι
    w : ι → 𝕜
    p : ι → E
    hf : ConvexOn 𝕜 s f
    h₀ : ∀ (i : ι), Membership.mem t i → LE.le 0 (w i)
    h₁ : LT.lt 0 (t.sum fun i => w i)
    hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
    hmem' : ∀ (i : ι), Membership.mem t i → Membership.mem (setOf fun p => And (Me …
    ⊢ LE.le (f (t.centerMass w p)) (t.centerMass w (Function.comp f p))
  -/
  convert (hf.convex_epigraph.centerMass_mem h₀ h₁ hmem').2 <;>
    /-
      case h.e'_3.h.e'_1
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_4
      ι : Type u_5
      inst✝⁵ : LinearOrderedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : OrderedAddCommGroup β
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 β
      inst✝ : OrderedSMul 𝕜 β
      s : Set E
      f : E → β
      t : Finset ι
      w : ι → 𝕜
      p : ι → E
      hf : ConvexOn 𝕜 s f
      h₀ : ∀ (i : ι), Membership.mem t i → LE.le 0 (w i)
      h₁ : LT.lt 0 (t.sum fun i => w i)
      hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
      hmem' : ∀ (i : ι), Membership.mem t i → Membership.mem (setOf fun p => And (Me …
      ⊢ Eq (t.centerMass w p) (t.centerMass w fun i => { fst := p i, snd := Function …
    -/
    /-
      🎉 no goals
    -/
    simp only [centerMass, Function.comp, Prod.smul_fst, Prod.fst_sum, Prod.smul_snd, Prod.snd_sum]
    /-
      🎉 no goals
    -/


/-- Concave **Jensen's inequality**, `Finset.centerMass` version. -/
theorem ConcaveOn.le_map_centerMass (hf : ConcaveOn 𝕜 s f) (h₀ : ∀ i ∈ t, 0 ≤ w i)
    (h₁ : 0 < ∑ i ∈ t, w i) (hmem : ∀ i ∈ t, p i ∈ s) :
    t.centerMass w (f ∘ p) ≤ f (t.centerMass w p) :=
  ConvexOn.map_centerMass_le (β := βᵒᵈ) hf h₀ h₁ hmem


/-- Convex **Jensen's inequality**, `Finset.sum` version. -/
theorem ConvexOn.map_sum_le (hf : ConvexOn 𝕜 s f) (h₀ : ∀ i ∈ t, 0 ≤ w i) (h₁ : ∑ i ∈ t, w i = 1)
    (hmem : ∀ i ∈ t, p i ∈ s) : f (∑ i ∈ t, w i • p i) ≤ ∑ i ∈ t, w i • f (p i) := by
  simpa only [centerMass, h₁, inv_one, one_smul] using
    hf.map_centerMass_le h₀ (h₁.symm ▸ zero_lt_one) hmem


/-- Concave **Jensen's inequality**, `Finset.sum` version. -/
theorem ConcaveOn.le_map_sum (hf : ConcaveOn 𝕜 s f) (h₀ : ∀ i ∈ t, 0 ≤ w i)
    (h₁ : ∑ i ∈ t, w i = 1) (hmem : ∀ i ∈ t, p i ∈ s) :
    (∑ i ∈ t, w i • f (p i)) ≤ f (∑ i ∈ t, w i • p i) :=
  ConvexOn.map_sum_le (β := βᵒᵈ) hf h₀ h₁ hmem


/-- Convex **Jensen's inequality** where an element plays a distinguished role. -/
lemma ConvexOn.map_add_sum_le (hf : ConvexOn 𝕜 s f) (h₀ : ∀ i ∈ t, 0 ≤ w i)
    (h₁ : v + ∑ i ∈ t, w i = 1) (hmem : ∀ i ∈ t, p i ∈ s) (hv : 0 ≤ v) (hq : q ∈ s) :
    f (v • q + ∑ i ∈ t, w i • p i) ≤ v • f q + ∑ i ∈ t, w i • f (p i) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    ι : Type u_5
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : OrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    t : Finset ι
    w : ι → 𝕜
    p : ι → E
    v : 𝕜
    q : E
    hf : ConvexOn 𝕜 s f
    h₀ : ∀ (i : ι), Membership.mem t i → LE.le 0 (w i)
    h₁ : Eq (HAdd.hAdd v (t.sum fun i => w i)) 1
    hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
    hv : LE.le 0 v
    hq : Membership.mem s q
    ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul v q) (t.sum fun i => HSMul.hSMul (w i) (p i …
  -/
  let W j := Option.elim j v w
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    ι : Type u_5
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : OrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    t : Finset ι
    w : ι → 𝕜
    p : ι → E
    v : 𝕜
    q : E
    hf : ConvexOn 𝕜 s f
    h₀ : ∀ (i : ι), Membership.mem t i → LE.le 0 (w i)
    h₁ : Eq (HAdd.hAdd v (t.sum fun i => w i)) 1
    hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
    hv : LE.le 0 v
    hq : Membership.mem s q
    W : Option ι → 𝕜 := fun j => j.elim v w
    ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul v q) (t.sum fun i => HSMul.hSMul (w i) (p i …
  -/
  let P j := Option.elim j q p
  have : f (∑ j ∈ insertNone t, W j • P j) ≤ ∑ j ∈ insertNone t, W j • f (P j) :=
    hf.map_sum_le (forall_mem_insertNone.2 ⟨hv, h₀⟩) (by simpa using h₁)
      (forall_mem_insertNone.2 ⟨hq, hmem⟩)
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    ι : Type u_5
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : OrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    t : Finset ι
    w : ι → 𝕜
    p : ι → E
    v : 𝕜
    q : E
    hf : ConvexOn 𝕜 s f
    h₀ : ∀ (i : ι), Membership.mem t i → LE.le 0 (w i)
    h₁ : Eq (HAdd.hAdd v (t.sum fun i => w i)) 1
    hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
    hv : LE.le 0 v
    hq : Membership.mem s q
    W : Option ι → 𝕜 := fun j => j.elim v w
    P : Option ι → E := fun j => j.elim q p
    this : LE.le (f ((Finset.insertNone t).sum fun j => HSMul.hSMul (W j) (P j)))  …
    ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul v q) (t.sum fun i => HSMul.hSMul (w i) (p i …
  -/
  simpa using this
  /-
    🎉 no goals
  -/


/-- Concave **Jensen's inequality** where an element plays a distinguished role. -/
lemma ConcaveOn.map_add_sum_le (hf : ConcaveOn 𝕜 s f) (h₀ : ∀ i ∈ t, 0 ≤ w i)
    (h₁ : v + ∑ i ∈ t, w i = 1) (hmem : ∀ i ∈ t, p i ∈ s) (hv : 0 ≤ v) (hq : q ∈ s) :
    v • f q + ∑ i ∈ t, w i • f (p i) ≤ f (v • q + ∑ i ∈ t, w i • p i) :=
  hf.dual.map_add_sum_le h₀ h₁ hmem hv hq


/-- Convex **strict Jensen inequality**.

If the function is strictly convex, the weights are strictly positive and the indexed family of
points is non-constant, then Jensen's inequality is strict.

See also `StrictConvexOn.map_sum_eq_iff`. -/
lemma StrictConvexOn.map_sum_lt (hf : StrictConvexOn 𝕜 s f) (h₀ : ∀ i ∈ t, 0 < w i)
    (h₁ : ∑ i ∈ t, w i = 1) (hmem : ∀ i ∈ t, p i ∈ s) (hp : ∃ j ∈ t, ∃ k ∈ t, p j ≠ p k) :
    f (∑ i ∈ t, w i • p i) < ∑ i ∈ t, w i • f (p i) := by
  classical
  obtain ⟨j, hj, k, hk, hjk⟩ := hp
  -- We replace `t` by `t \ {j, k}`
  have : k ∈ t.erase j := mem_erase.2 ⟨ne_of_apply_ne _ hjk.symm, hk⟩
  let u := (t.erase j).erase k
  have hj : j ∉ u := by simp [u]
  have hk : k ∉ u := by simp [u]
  have ht :
      t = (u.cons k hk).cons j (mem_cons.not.2 <| not_or_intro (ne_of_apply_ne _ hjk) hj) := by
    simp [u, insert_erase this, insert_erase ‹j ∈ t›, *]
  clear_value u
  subst ht
  simp only [sum_cons]
  have := h₀ j <| by simp
  have := h₀ k <| by simp
  let c := w j + w k
  have hc : w j / c + w k / c = 1 := by field_simp [c]
  calc f (w j • p j + (w k • p k + ∑ x ∈ u, w x • p x))
    _ = f (c • ((w j / c) • p j + (w k / c) • p k) + ∑ x ∈ u, w x • p x) := by
      congrm f ?_
      match_scalars <;> field_simp
    _ ≤ c • f ((w j / c) • p j + (w k / c) • p k) + ∑ x ∈ u, w x • f (p x) :=
      -- apply the usual Jensen's inequality wrt the weighted average of the two distinguished
      -- points and all the other points
        hf.convexOn.map_add_sum_le (fun i hi ↦ (h₀ _ <| by simp [hi]).le)
          (by simpa [-cons_eq_insert, ← add_assoc] using h₁)
          (forall_of_forall_cons <| forall_of_forall_cons hmem) (by positivity) <| by
           refine hf.1 (hmem _ <| by simp) (hmem _ <| by simp) ?_ ?_ hc <;> positivity
    _ < c • ((w j / c) • f (p j) + (w k / c) • f (p k)) + ∑ x ∈ u, w x • f (p x) := by
      -- then apply the definition of strict convexity for the two distinguished points
      gcongr; refine hf.2 (hmem _ <| by simp) (hmem _ <| by simp) hjk ?_ ?_ hc <;> positivity
    _ = (w j • f (p j) + w k • f (p k)) + ∑ x ∈ u, w x • f (p x) := by
      match_scalars <;> field_simp
    _ = w j • f (p j) + (w k • f (p k) + ∑ x ∈ u, w x • f (p x)) := by abel_nf


/-- Concave **strict Jensen inequality**.

If the function is strictly concave, the weights are strictly positive and the indexed family of
points is non-constant, then Jensen's inequality is strict.

See also `StrictConcaveOn.map_sum_eq_iff`. -/
lemma StrictConcaveOn.lt_map_sum (hf : StrictConcaveOn 𝕜 s f) (h₀ : ∀ i ∈ t, 0 < w i)
    (h₁ : ∑ i ∈ t, w i = 1) (hmem : ∀ i ∈ t, p i ∈ s) (hp : ∃ j ∈ t, ∃ k ∈ t, p j ≠ p k) :
    ∑ i ∈ t, w i • f (p i) < f (∑ i ∈ t, w i • p i) := hf.dual.map_sum_lt h₀ h₁ hmem hp


/-- A form of the **equality case of Jensen's equality**.

For a strictly convex function `f` and positive weights `w`, if
`f (∑ i ∈ t, w i • p i) = ∑ i ∈ t, w i • f (p i)`, then the points `p` are all equal.

See also `StrictConvexOn.map_sum_eq_iff`. -/
lemma StrictConvexOn.eq_of_le_map_sum (hf : StrictConvexOn 𝕜 s f) (h₀ : ∀ i ∈ t, 0 < w i)
    (h₁ : ∑ i ∈ t, w i = 1) (hmem : ∀ i ∈ t, p i ∈ s)
    (h_eq : ∑ i ∈ t, w i • f (p i) ≤ f (∑ i ∈ t, w i • p i)) :
    ∀ ⦃j⦄, j ∈ t → ∀ ⦃k⦄, k ∈ t → p j = p k := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    ι : Type u_5
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : OrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    t : Finset ι
    w : ι → 𝕜
    p : ι → E
    hf : StrictConvexOn 𝕜 s f
    h₀ : ∀ (i : ι), Membership.mem t i → LT.lt 0 (w i)
    h₁ : Eq (t.sum fun i => w i) 1
    hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
    h_eq : LE.le (t.sum fun i => HSMul.hSMul (w i) (f (p i))) (f (t.sum fun i => H …
    ⊢ ∀ ⦃j : ι⦄, Membership.mem t j → ∀ ⦃k : ι⦄, Membership.mem t k → Eq (p j) (p k)
  -/
  by_contra!; exact h_eq.not_lt <| hf.map_sum_lt h₀ h₁ hmem this
              /-
                🎉 no goals
              -/


/-- A form of the **equality case of Jensen's equality**.

For a strictly concave function `f` and positive weights `w`, if
`f (∑ i ∈ t, w i • p i) = ∑ i ∈ t, w i • f (p i)`, then the points `p` are all equal.

See also `StrictConcaveOn.map_sum_eq_iff`. -/
lemma StrictConcaveOn.eq_of_map_sum_eq (hf : StrictConcaveOn 𝕜 s f) (h₀ : ∀ i ∈ t, 0 < w i)
    (h₁ : ∑ i ∈ t, w i = 1) (hmem : ∀ i ∈ t, p i ∈ s)
    (h_eq : f (∑ i ∈ t, w i • p i) ≤ ∑ i ∈ t, w i • f (p i)) :
    ∀ ⦃j⦄, j ∈ t → ∀ ⦃k⦄, k ∈ t → p j = p k := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    ι : Type u_5
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : OrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    t : Finset ι
    w : ι → 𝕜
    p : ι → E
    hf : StrictConcaveOn 𝕜 s f
    h₀ : ∀ (i : ι), Membership.mem t i → LT.lt 0 (w i)
    h₁ : Eq (t.sum fun i => w i) 1
    hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
    h_eq : LE.le (f (t.sum fun i => HSMul.hSMul (w i) (p i))) (t.sum fun i => HSMu …
    ⊢ ∀ ⦃j : ι⦄, Membership.mem t j → ∀ ⦃k : ι⦄, Membership.mem t k → Eq (p j) (p k)
  -/
  by_contra!; exact h_eq.not_lt <| hf.lt_map_sum h₀ h₁ hmem this
              /-
                🎉 no goals
              -/


/-- Canonical form of the **equality case of Jensen's equality**.

For a strictly convex function `f` and positive weights `w`, we have
`f (∑ i ∈ t, w i • p i) = ∑ i ∈ t, w i • f (p i)` if and only if the points `p` are all equal
(and in fact all equal to their center of mass wrt `w`). -/
lemma StrictConvexOn.map_sum_eq_iff {w : ι → 𝕜} {p : ι → E} (hf : StrictConvexOn 𝕜 s f)
    (h₀ : ∀ i ∈ t, 0 < w i) (h₁ : ∑ i ∈ t, w i = 1) (hmem : ∀ i ∈ t, p i ∈ s) :
    f (∑ i ∈ t, w i • p i) = ∑ i ∈ t, w i • f (p i) ↔ ∀ j ∈ t, p j = ∑ i ∈ t, w i • p i := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    ι : Type u_5
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : OrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    t : Finset ι
    w : ι → 𝕜
    p : ι → E
    hf : StrictConvexOn 𝕜 s f
    h₀ : ∀ (i : ι), Membership.mem t i → LT.lt 0 (w i)
    h₁ : Eq (t.sum fun i => w i) 1
    hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
    ⊢ Iff (Eq (f (t.sum fun i => HSMul.hSMul (w i) (p i))) (t.sum fun i => HSMul.h …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_4
      ι : Type u_5
      inst✝⁵ : LinearOrderedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : OrderedAddCommGroup β
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 β
      inst✝ : OrderedSMul 𝕜 β
      s : Set E
      f : E → β
      t : Finset ι
      w : ι → 𝕜
      p : ι → E
      hf : StrictConvexOn 𝕜 s f
      h₀ : ∀ (i : ι), Membership.mem t i → LT.lt 0 (w i)
      h₁ : Eq (t.sum fun i => w i) 1
      hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
      ⊢ Eq (f (t.sum fun i => HSMul.hSMul (w i) (p i))) (t.sum fun i => HSMul.hSMul  …
    -/
  · obtain rfl | ⟨i₀, hi₀⟩ := t.eq_empty_or_nonempty
      /-
        case mp.inl
        𝕜 : Type u_1
        E : Type u_2
        β : Type u_4
        ι : Type u_5
        inst✝⁵ : LinearOrderedField 𝕜
        inst✝⁴ : AddCommGroup E
        inst✝³ : OrderedAddCommGroup β
        inst✝² : Module 𝕜 E
        inst✝¹ : Module 𝕜 β
        inst✝ : OrderedSMul 𝕜 β
        s : Set E
        f : E → β
        w : ι → 𝕜
        p : ι → E
        hf : StrictConvexOn 𝕜 s f
        h₀ : ∀ (i : ι), Membership.mem EmptyCollection.emptyCollection i → LT.lt 0 (w i)
        h₁ : Eq (EmptyCollection.emptyCollection.sum fun i => w i) 1
        hmem : ∀ (i : ι), Membership.mem EmptyCollection.emptyCollection i → Membershi …
        ⊢ Eq (f (EmptyCollection.emptyCollection.sum fun i => HSMul.hSMul (w i) (p i)) …
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case mp.inr.intro
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_4
      ι : Type u_5
      inst✝⁵ : LinearOrderedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : OrderedAddCommGroup β
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 β
      inst✝ : OrderedSMul 𝕜 β
      s : Set E
      f : E → β
      t : Finset ι
      w : ι → 𝕜
      p : ι → E
      hf : StrictConvexOn 𝕜 s f
      h₀ : ∀ (i : ι), Membership.mem t i → LT.lt 0 (w i)
      h₁ : Eq (t.sum fun i => w i) 1
      hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
      i₀ : ι
      hi₀ : Membership.mem t i₀
      ⊢ Eq (f (t.sum fun i => HSMul.hSMul (w i) (p i))) (t.sum fun i => HSMul.hSMul  …
    -/
    intro h_eq i hi
    have H : ∀ j ∈ t, p j = p i₀ := by
      intro j hj
      apply hf.eq_of_le_map_sum h₀ h₁ hmem h_eq.ge hj hi₀
    calc p i = p i₀ := by rw [H _ hi]
      _ = (1 : 𝕜) • p i₀ := by simp
      _ = (∑ j ∈ t, w j) • p i₀ := by rw [h₁]
      _ = ∑ j ∈ t, (w j • p i₀) := by rw [sum_smul]
      _ = ∑ j ∈ t, (w j • p j) := by congr! 2 with j hj; rw [← H _ hj]
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_4
      ι : Type u_5
      inst✝⁵ : LinearOrderedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : OrderedAddCommGroup β
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 β
      inst✝ : OrderedSMul 𝕜 β
      s : Set E
      f : E → β
      t : Finset ι
      w : ι → 𝕜
      p : ι → E
      hf : StrictConvexOn 𝕜 s f
      h₀ : ∀ (i : ι), Membership.mem t i → LT.lt 0 (w i)
      h₁ : Eq (t.sum fun i => w i) 1
      hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
      ⊢ (∀ (j : ι), Membership.mem t j → Eq (p j) (t.sum fun i => HSMul.hSMul (w i)  …
    -/
  · intro h
    have H : ∀ j ∈ t, w j • f (p j) = w j • f (∑ i ∈ t, w i • p i) := by
      intro j hj
      simp [h j hj]
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_4
      ι : Type u_5
      inst✝⁵ : LinearOrderedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : OrderedAddCommGroup β
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 β
      inst✝ : OrderedSMul 𝕜 β
      s : Set E
      f : E → β
      t : Finset ι
      w : ι → 𝕜
      p : ι → E
      hf : StrictConvexOn 𝕜 s f
      h₀ : ∀ (i : ι), Membership.mem t i → LT.lt 0 (w i)
      h₁ : Eq (t.sum fun i => w i) 1
      hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
      h : ∀ (j : ι), Membership.mem t j → Eq (p j) (t.sum fun i => HSMul.hSMul (w i) …
      H : ∀ (j : ι), Membership.mem t j → Eq (HSMul.hSMul (w j) (f (p j))) (HSMul.hS …
      ⊢ Eq (f (t.sum fun i => HSMul.hSMul (w i) (p i))) (t.sum fun i => HSMul.hSMul  …
    -/
    rw [sum_congr rfl H, ← sum_smul, h₁, one_smul]
    /-
      🎉 no goals
    -/


/-- Canonical form of the **equality case of Jensen's equality**.

For a strictly concave function `f` and positive weights `w`, we have
`f (∑ i ∈ t, w i • p i) = ∑ i ∈ t, w i • f (p i)` if and only if the points `p` are all equal
(and in fact all equal to their center of mass wrt `w`). -/
lemma StrictConcaveOn.map_sum_eq_iff (hf : StrictConcaveOn 𝕜 s f) (h₀ : ∀ i ∈ t, 0 < w i)
    (h₁ : ∑ i ∈ t, w i = 1) (hmem : ∀ i ∈ t, p i ∈ s) :
    f (∑ i ∈ t, w i • p i) = ∑ i ∈ t, w i • f (p i) ↔ ∀ j ∈ t, p j = ∑ i ∈ t, w i • p i := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    ι : Type u_5
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : OrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    t : Finset ι
    w : ι → 𝕜
    p : ι → E
    hf : StrictConcaveOn 𝕜 s f
    h₀ : ∀ (i : ι), Membership.mem t i → LT.lt 0 (w i)
    h₁ : Eq (t.sum fun i => w i) 1
    hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
    ⊢ Iff (Eq (f (t.sum fun i => HSMul.hSMul (w i) (p i))) (t.sum fun i => HSMul.h …
  -/
  simpa using hf.neg.map_sum_eq_iff h₀ h₁ hmem
  /-
    🎉 no goals
  -/


/-- Canonical form of the **equality case of Jensen's equality**.

For a strictly convex function `f` and nonnegative weights `w`, we have
`f (∑ i ∈ t, w i • p i) = ∑ i ∈ t, w i • f (p i)` if and only if the points `p` with nonzero
weight are all equal (and in fact all equal to their center of mass wrt `w`). -/
lemma StrictConvexOn.map_sum_eq_iff' (hf : StrictConvexOn 𝕜 s f) (h₀ : ∀ i ∈ t, 0 ≤ w i)
    (h₁ : ∑ i ∈ t, w i = 1) (hmem : ∀ i ∈ t, p i ∈ s) :
    f (∑ i ∈ t, w i • p i) = ∑ i ∈ t, w i • f (p i) ↔
      ∀ j ∈ t, w j ≠ 0 → p j = ∑ i ∈ t, w i • p i := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    ι : Type u_5
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : OrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    t : Finset ι
    w : ι → 𝕜
    p : ι → E
    hf : StrictConvexOn 𝕜 s f
    h₀ : ∀ (i : ι), Membership.mem t i → LE.le 0 (w i)
    h₁ : Eq (t.sum fun i => w i) 1
    hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
    ⊢ Iff (Eq (f (t.sum fun i => HSMul.hSMul (w i) (p i))) (t.sum fun i => HSMul.h …
  -/
  have hw (i) (_ : i ∈ t) : w i • p i ≠ 0 → w i ≠ 0 := by aesop
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    ι : Type u_5
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : OrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    t : Finset ι
    w : ι → 𝕜
    p : ι → E
    hf : StrictConvexOn 𝕜 s f
    h₀ : ∀ (i : ι), Membership.mem t i → LE.le 0 (w i)
    h₁ : Eq (t.sum fun i => w i) 1
    hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
    hw : ∀ (i : ι), Membership.mem t i → Ne (HSMul.hSMul (w i) (p i)) 0 → Ne (w i) 0
    ⊢ Iff (Eq (f (t.sum fun i => HSMul.hSMul (w i) (p i))) (t.sum fun i => HSMul.h …
  -/
  have hw' (i) (_ : i ∈ t) : w i • f (p i) ≠ 0 → w i ≠ 0 := by aesop
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    ι : Type u_5
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : OrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    t : Finset ι
    w : ι → 𝕜
    p : ι → E
    hf : StrictConvexOn 𝕜 s f
    h₀ : ∀ (i : ι), Membership.mem t i → LE.le 0 (w i)
    h₁ : Eq (t.sum fun i => w i) 1
    hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
    hw : ∀ (i : ι), Membership.mem t i → Ne (HSMul.hSMul (w i) (p i)) 0 → Ne (w i) 0
    hw' : ∀ (i : ι), Membership.mem t i → Ne (HSMul.hSMul (w i) (f (p i))) 0 → Ne  …
    ⊢ Iff (Eq (f (t.sum fun i => HSMul.hSMul (w i) (p i))) (t.sum fun i => HSMul.h …
  -/
  rw [← sum_filter_of_ne hw, ← sum_filter_of_ne hw', hf.map_sum_eq_iff]
    /-
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_4
      ι : Type u_5
      inst✝⁵ : LinearOrderedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : OrderedAddCommGroup β
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 β
      inst✝ : OrderedSMul 𝕜 β
      s : Set E
      f : E → β
      t : Finset ι
      w : ι → 𝕜
      p : ι → E
      hf : StrictConvexOn 𝕜 s f
      h₀ : ∀ (i : ι), Membership.mem t i → LE.le 0 (w i)
      h₁ : Eq (t.sum fun i => w i) 1
      hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
      hw : ∀ (i : ι), Membership.mem t i → Ne (HSMul.hSMul (w i) (p i)) 0 → Ne (w i) 0
      hw' : ∀ (i : ι), Membership.mem t i → Ne (HSMul.hSMul (w i) (f (p i))) 0 → Ne  …
      ⊢ Iff (∀ (j : ι), Membership.mem (Finset.filter (fun x => Ne (w x) 0) t) j → E …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h₀
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_4
      ι : Type u_5
      inst✝⁵ : LinearOrderedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : OrderedAddCommGroup β
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 β
      inst✝ : OrderedSMul 𝕜 β
      s : Set E
      f : E → β
      t : Finset ι
      w : ι → 𝕜
      p : ι → E
      hf : StrictConvexOn 𝕜 s f
      h₀ : ∀ (i : ι), Membership.mem t i → LE.le 0 (w i)
      h₁ : Eq (t.sum fun i => w i) 1
      hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
      hw : ∀ (i : ι), Membership.mem t i → Ne (HSMul.hSMul (w i) (p i)) 0 → Ne (w i) 0
      hw' : ∀ (i : ι), Membership.mem t i → Ne (HSMul.hSMul (w i) (f (p i))) 0 → Ne  …
      ⊢ ∀ (i : ι), Membership.mem (Finset.filter (fun x => Ne (w x) 0) t) i → LT.lt  …
    -/
  · simp +contextual [(h₀ _ _).gt_iff_ne]
    /-
      🎉 no goals
    -/
    /-
      case h₁
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_4
      ι : Type u_5
      inst✝⁵ : LinearOrderedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : OrderedAddCommGroup β
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 β
      inst✝ : OrderedSMul 𝕜 β
      s : Set E
      f : E → β
      t : Finset ι
      w : ι → 𝕜
      p : ι → E
      hf : StrictConvexOn 𝕜 s f
      h₀ : ∀ (i : ι), Membership.mem t i → LE.le 0 (w i)
      h₁ : Eq (t.sum fun i => w i) 1
      hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
      hw : ∀ (i : ι), Membership.mem t i → Ne (HSMul.hSMul (w i) (p i)) 0 → Ne (w i) 0
      hw' : ∀ (i : ι), Membership.mem t i → Ne (HSMul.hSMul (w i) (f (p i))) 0 → Ne  …
      ⊢ Eq ((Finset.filter (fun x => Ne (w x) 0) t).sum fun i => w i) 1
    -/
  · rwa [sum_filter_ne_zero]
    /-
      🎉 no goals
    -/
    /-
      case hmem
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_4
      ι : Type u_5
      inst✝⁵ : LinearOrderedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : OrderedAddCommGroup β
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 β
      inst✝ : OrderedSMul 𝕜 β
      s : Set E
      f : E → β
      t : Finset ι
      w : ι → 𝕜
      p : ι → E
      hf : StrictConvexOn 𝕜 s f
      h₀ : ∀ (i : ι), Membership.mem t i → LE.le 0 (w i)
      h₁ : Eq (t.sum fun i => w i) 1
      hmem : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
      hw : ∀ (i : ι), Membership.mem t i → Ne (HSMul.hSMul (w i) (p i)) 0 → Ne (w i) 0
      hw' : ∀ (i : ι), Membership.mem t i → Ne (HSMul.hSMul (w i) (f (p i))) 0 → Ne  …
      ⊢ ∀ (i : ι), Membership.mem (Finset.filter (fun x => Ne (w x) 0) t) i → Member …
    -/
  · simp +contextual [hmem _ _]
    /-
      🎉 no goals
    -/


/-- Canonical form of the **equality case of Jensen's equality**.

For a strictly concave function `f` and nonnegative weights `w`, we have
`f (∑ i ∈ t, w i • p i) = ∑ i ∈ t, w i • f (p i)` if and only if the points `p` with nonzero
weight are all equal (and in fact all equal to their center of mass wrt `w`). -/
lemma StrictConcaveOn.map_sum_eq_iff' (hf : StrictConcaveOn 𝕜 s f) (h₀ : ∀ i ∈ t, 0 ≤ w i)
    (h₁ : ∑ i ∈ t, w i = 1) (hmem : ∀ i ∈ t, p i ∈ s) :
    f (∑ i ∈ t, w i • p i) = ∑ i ∈ t, w i • f (p i) ↔
      ∀ j ∈ t, w j ≠ 0 → p j = ∑ i ∈ t, w i • p i := hf.dual.map_sum_eq_iff' h₀ h₁ hmem


theorem ConvexOn.le_sup_of_mem_convexHull {t : Finset E} (hf : ConvexOn 𝕜 s f) (hts : ↑t ⊆ s)
    (hx : x ∈ convexHull 𝕜 (t : Set E)) :
    f x ≤ t.sup' (coe_nonempty.1 <| convexHull_nonempty_iff.1 ⟨x, hx⟩) f := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : LinearOrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    x : E
    t : Finset E
    hf : ConvexOn 𝕜 s f
    hts : HasSubset.Subset (↑t) s
    hx : Membership.mem ((convexHull 𝕜) ↑t) x
    ⊢ LE.le (f x) (t.sup' ⋯ f)
  -/
  obtain ⟨w, hw₀, hw₁, rfl⟩ := mem_convexHull.1 hx
  exact (hf.map_centerMass_le hw₀ (by positivity) hts).trans
    (centerMass_le_sup hw₀ <| by positivity)


theorem ConvexOn.inf_le_of_mem_convexHull {t : Finset E} (hf : ConcaveOn 𝕜 s f) (hts : ↑t ⊆ s)
    (hx : x ∈ convexHull 𝕜 (t : Set E)) :
    t.inf' (coe_nonempty.1 <| convexHull_nonempty_iff.1 ⟨x, hx⟩) f ≤ f x :=
  hf.dual.le_sup_of_mem_convexHull hts hx


@[deprecated (since := "2024-08-25")]
alias le_sup_of_mem_convexHull := ConvexOn.le_sup_of_mem_convexHull


@[deprecated (since := "2024-08-25")]
alias inf_le_of_mem_convexHull := ConvexOn.inf_le_of_mem_convexHull


/-- If a function `f` is convex on `s`, then the value it takes at some center of mass of points of
`s` is less than the value it takes on one of those points. -/
lemma ConvexOn.exists_ge_of_centerMass {t : Finset ι} (h : ConvexOn 𝕜 s f)
    (hw₀ : ∀ i ∈ t, 0 ≤ w i) (hw₁ : 0 < ∑ i ∈ t, w i) (hp : ∀ i ∈ t, p i ∈ s) :
    ∃ i ∈ t, f (t.centerMass w p) ≤ f (p i) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    ι : Type u_5
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : LinearOrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    w : ι → 𝕜
    p : ι → E
    t : Finset ι
    h : ConvexOn 𝕜 s f
    hw₀ : ∀ (i : ι), Membership.mem t i → LE.le 0 (w i)
    hw₁ : LT.lt 0 (t.sum fun i => w i)
    hp : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
    ⊢ Exists fun i => And (Membership.mem t i) (LE.le (f (t.centerMass w p)) (f (p …
  -/
  set y := t.centerMass w p
  -- TODO: can `rsuffices` be used to write the `exact` first, then the proof of this obtain?
  obtain ⟨i, hi, hfi⟩ : ∃ i ∈ {i ∈ t | w i ≠ 0}, w i • f y ≤ w i • (f ∘ p) i := by
    have hw' : (0 : 𝕜) < ∑ i ∈ t with w i ≠ 0, w i := by rwa [sum_filter_ne_zero]
    refine exists_le_of_sum_le (nonempty_of_sum_ne_zero hw'.ne') ?_
    rw [← sum_smul, ← smul_le_smul_iff_of_pos_left (inv_pos.2 hw'), inv_smul_smul₀ hw'.ne', ←
      centerMass, centerMass_filter_ne_zero]
    exact h.map_centerMass_le hw₀ hw₁ hp
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    ι : Type u_5
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : LinearOrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    w : ι → 𝕜
    p : ι → E
    t : Finset ι
    h : ConvexOn 𝕜 s f
    hw₀ : ∀ (i : ι), Membership.mem t i → LE.le 0 (w i)
    hw₁ : LT.lt 0 (t.sum fun i => w i)
    hp : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
    y : E := t.centerMass w p
    i : ι
    hi : Membership.mem (Finset.filter (fun i => Ne (w i) 0) t) i
    hfi : LE.le (HSMul.hSMul (w i) (f y)) (HSMul.hSMul (w i) (Function.comp f p i))
    ⊢ Exists fun i => And (Membership.mem t i) (LE.le (f y) (f (p i)))
  -/
  rw [mem_filter] at hi
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    ι : Type u_5
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : LinearOrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    w : ι → 𝕜
    p : ι → E
    t : Finset ι
    h : ConvexOn 𝕜 s f
    hw₀ : ∀ (i : ι), Membership.mem t i → LE.le 0 (w i)
    hw₁ : LT.lt 0 (t.sum fun i => w i)
    hp : ∀ (i : ι), Membership.mem t i → Membership.mem s (p i)
    y : E := t.centerMass w p
    i : ι
    hi : And (Membership.mem t i) (Ne (w i) 0)
    hfi : LE.le (HSMul.hSMul (w i) (f y)) (HSMul.hSMul (w i) (Function.comp f p i))
    ⊢ Exists fun i => And (Membership.mem t i) (LE.le (f y) (f (p i)))
  -/
  exact ⟨i, hi.1, (smul_le_smul_iff_of_pos_left <| (hw₀ i hi.1).lt_of_ne hi.2.symm).1 hfi⟩
  /-
    🎉 no goals
  -/


/-- If a function `f` is concave on `s`, then the value it takes at some center of mass of points of
`s` is greater than the value it takes on one of those points. -/
lemma ConcaveOn.exists_le_of_centerMass {t : Finset ι} (h : ConcaveOn 𝕜 s f)
    (hw₀ : ∀ i ∈ t, 0 ≤ w i) (hw₁ : 0 < ∑ i ∈ t, w i) (hp : ∀ i ∈ t, p i ∈ s) :
    ∃ i ∈ t, f (p i) ≤ f (t.centerMass w p) := h.dual.exists_ge_of_centerMass hw₀ hw₁ hp


/-- **Maximum principle** for convex functions. If a function `f` is convex on the convex hull of
`s`, then the eventual maximum of `f` on `convexHull 𝕜 s` lies in `s`. -/
lemma ConvexOn.exists_ge_of_mem_convexHull {t : Set E} (hf : ConvexOn 𝕜 s f) (hts : t ⊆ s)
    (hx : x ∈ convexHull 𝕜 t) : ∃ y ∈ t, f x ≤ f y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : LinearOrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    x : E
    t : Set E
    hf : ConvexOn 𝕜 s f
    hts : HasSubset.Subset t s
    hx : Membership.mem ((convexHull 𝕜) t) x
    ⊢ Exists fun y => And (Membership.mem t y) (LE.le (f x) (f y))
  -/
  rw [_root_.convexHull_eq] at hx
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : LinearOrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    x : E
    t : Set E
    hf : ConvexOn 𝕜 s f
    hts : HasSubset.Subset t s
    hx : Membership.mem (setOf fun x => Exists fun ι => Exists fun t_1 => Exists f …
    ⊢ Exists fun y => And (Membership.mem t y) (LE.le (f x) (f y))
  -/
  obtain ⟨α, t, w, p, hw₀, hw₁, hp, rfl⟩ := hx
  obtain ⟨i, hit, Hi⟩ := hf.exists_ge_of_centerMass hw₀ (hw₁.symm ▸ zero_lt_one)
    fun i hi ↦ hts (hp i hi)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : LinearOrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    t✝ : Set E
    hf : ConvexOn 𝕜 s f
    hts : HasSubset.Subset t✝ s
    α : Type
    t : Finset α
    w : α → 𝕜
    p : α → E
    hw₀ : ∀ (i : α), Membership.mem t i → LE.le 0 (w i)
    hw₁ : Eq (t.sum fun i => w i) 1
    hp : ∀ (i : α), Membership.mem t i → Membership.mem t✝ (p i)
    i : α
    hit : Membership.mem t i
    Hi : LE.le (f (t.centerMass w p)) (f (p i))
    ⊢ Exists fun y => And (Membership.mem t✝ y) (LE.le (f (t.centerMass w p)) (f y))
  -/
  exact ⟨p i, hp i hit, Hi⟩
  /-
    🎉 no goals
  -/


/-- **Minimum principle** for concave functions. If a function `f` is concave on the convex hull of
`s`, then the eventual minimum of `f` on `convexHull 𝕜 s` lies in `s`. -/
lemma ConcaveOn.exists_le_of_mem_convexHull {t : Set E} (hf : ConcaveOn 𝕜 s f) (hts : t ⊆ s)
    (hx : x ∈ convexHull 𝕜 t) : ∃ y ∈ t, f y ≤ f x := hf.dual.exists_ge_of_mem_convexHull hts hx


/-- **Maximum principle** for convex functions on a segment. If a function `f` is convex on the
segment `[x, y]`, then the eventual maximum of `f` on `[x, y]` is at `x` or `y`. -/
lemma ConvexOn.le_max_of_mem_segment (hf : ConvexOn 𝕜 s f) (hx : x ∈ s) (hy : y ∈ s)
    (hz : z ∈ [x -[𝕜] y]) : f z ≤ max (f x) (f y) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : LinearOrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    x y z : E
    hf : ConvexOn 𝕜 s f
    hx : Membership.mem s x
    hy : Membership.mem s y
    hz : Membership.mem (segment 𝕜 x y) z
    ⊢ LE.le (f z) (Max.max (f x) (f y))
  -/
  rw [← convexHull_pair] at hz; simpa using hf.exists_ge_of_mem_convexHull (pair_subset hx hy) hz
                                /-
                                  🎉 no goals
                                -/


/-- **Minimum principle** for concave functions on a segment. If a function `f` is concave on the
segment `[x, y]`, then the eventual minimum of `f` on `[x, y]` is at `x` or `y`. -/
lemma ConcaveOn.min_le_of_mem_segment (hf : ConcaveOn 𝕜 s f) (hx : x ∈ s) (hy : y ∈ s)
    (hz : z ∈ [x -[𝕜] y]) : min (f x) (f y) ≤ f z := hf.dual.le_max_of_mem_segment hx hy hz


/-- **Maximum principle** for convex functions on an interval. If a function `f` is convex on the
interval `[x, y]`, then the eventual maximum of `f` on `[x, y]` is at `x` or `y`. -/
lemma ConvexOn.le_max_of_mem_Icc {s : Set 𝕜} {f : 𝕜 → β} {x y z : 𝕜} (hf : ConvexOn 𝕜 s f)
    (hx : x ∈ s) (hy : y ∈ s) (hz : z ∈ Icc x y) : f z ≤ max (f x) (f y) := by
  /-
    𝕜 : Type u_1
    β : Type u_4
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : LinearOrderedAddCommGroup β
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set 𝕜
    f : 𝕜 → β
    x y z : 𝕜
    hf : ConvexOn 𝕜 s f
    hx : Membership.mem s x
    hy : Membership.mem s y
    hz : Membership.mem (Set.Icc x y) z
    ⊢ LE.le (f z) (Max.max (f x) (f y))
  -/
  rw [← segment_eq_Icc (hz.1.trans hz.2)] at hz; exact hf.le_max_of_mem_segment hx hy hz
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- **Minimum principle** for concave functions on an interval. If a function `f` is concave on the
interval `[x, y]`, then the eventual minimum of `f` on `[x, y]` is at `x` or `y`. -/
lemma ConcaveOn.min_le_of_mem_Icc {s : Set 𝕜} {f : 𝕜 → β} {x y z : 𝕜} (hf : ConcaveOn 𝕜 s f)
    (hx : x ∈ s) (hy : y ∈ s) (hz : z ∈ Icc x y) : min (f x) (f y) ≤ f z :=
  hf.dual.le_max_of_mem_Icc hx hy hz


lemma ConvexOn.bddAbove_convexHull {s t : Set E} (hst : s ⊆ t) (hf : ConvexOn 𝕜 t f) :
    BddAbove (f '' s) → BddAbove (f '' convexHull 𝕜 s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : LinearOrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    f : E → β
    s t : Set E
    hst : HasSubset.Subset s t
    hf : ConvexOn 𝕜 t f
    ⊢ BddAbove (Set.image f s) → BddAbove (Set.image f ((convexHull 𝕜) s))
  -/
  rintro ⟨b, hb⟩
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : LinearOrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    f : E → β
    s t : Set E
    hst : HasSubset.Subset s t
    hf : ConvexOn 𝕜 t f
    b : β
    hb : Membership.mem (upperBounds (Set.image f s)) b
    ⊢ BddAbove (Set.image f ((convexHull 𝕜) s))
  -/
  refine ⟨b, ?_⟩
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : LinearOrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    f : E → β
    s t : Set E
    hst : HasSubset.Subset s t
    hf : ConvexOn 𝕜 t f
    b : β
    hb : Membership.mem (upperBounds (Set.image f s)) b
    ⊢ Membership.mem (upperBounds (Set.image f ((convexHull 𝕜) s))) b
  -/
  rintro _ ⟨x, hx, rfl⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : LinearOrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    f : E → β
    s t : Set E
    hst : HasSubset.Subset s t
    hf : ConvexOn 𝕜 t f
    b : β
    hb : Membership.mem (upperBounds (Set.image f s)) b
    x : E
    hx : Membership.mem ((convexHull 𝕜) s) x
    ⊢ LE.le (f x) b
  -/
  obtain ⟨y, hy, hxy⟩ := hf.exists_ge_of_mem_convexHull hst hx
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : LinearOrderedAddCommGroup β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    f : E → β
    s t : Set E
    hst : HasSubset.Subset s t
    hf : ConvexOn 𝕜 t f
    b : β
    hb : Membership.mem (upperBounds (Set.image f s)) b
    x : E
    hx : Membership.mem ((convexHull 𝕜) s) x
    y : E
    hy : Membership.mem s y
    hxy : LE.le (f x) (f y)
    ⊢ LE.le (f x) b
  -/
  exact hxy.trans <| hb <| mem_image_of_mem _ hy
  /-
    🎉 no goals
  -/


lemma ConcaveOn.bddBelow_convexHull {s t : Set E} (hst : s ⊆ t) (hf : ConcaveOn 𝕜 t f) :
    BddBelow (f '' s) → BddBelow (f '' convexHull 𝕜 s) := hf.dual.bddAbove_convexHull hst


