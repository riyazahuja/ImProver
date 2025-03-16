/-- The largest balanced subset of `s`. -/
def balancedCore (s : Set E) :=
  ⋃₀ { t : Set E | Balanced 𝕜 t ∧ t ⊆ s }


/-- Helper definition to prove `balanced_core_eq_iInter`-/
def balancedCoreAux (s : Set E) :=
  ⋂ (r : 𝕜) (_ : 1 ≤ ‖r‖), r • s


/-- The smallest balanced superset of `s`. -/
def balancedHull (s : Set E) :=
  ⋃ (r : 𝕜) (_ : ‖r‖ ≤ 1), r • s


theorem balancedCore_subset (s : Set E) : balancedCore 𝕜 s ⊆ s :=
  sUnion_subset fun _ ht => ht.2


theorem balancedCore_empty : balancedCore 𝕜 (∅ : Set E) = ∅ :=
  eq_empty_of_subset_empty (balancedCore_subset _)


theorem mem_balancedCore_iff : x ∈ balancedCore 𝕜 s ↔ ∃ t, Balanced 𝕜 t ∧ t ⊆ s ∧ x ∈ t := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹ : SeminormedRing 𝕜
    inst✝ : SMul 𝕜 E
    s : Set E
    x : E
    ⊢ Iff (Membership.mem (balancedCore 𝕜 s) x) (Exists fun t => And (Balanced 𝕜 t …
  -/
  simp_rw [balancedCore, mem_sUnion, mem_setOf_eq, and_assoc]
  /-
    🎉 no goals
  -/


theorem smul_balancedCore_subset (s : Set E) {a : 𝕜} (ha : ‖a‖ ≤ 1) :
    a • balancedCore 𝕜 s ⊆ balancedCore 𝕜 s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹ : SeminormedRing 𝕜
    inst✝ : SMul 𝕜 E
    s : Set E
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    ⊢ HasSubset.Subset (HSMul.hSMul a (balancedCore 𝕜 s)) (balancedCore 𝕜 s)
  -/
  rintro x ⟨y, hy, rfl⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹ : SeminormedRing 𝕜
    inst✝ : SMul 𝕜 E
    s : Set E
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    y : E
    hy : Membership.mem (balancedCore 𝕜 s) y
    ⊢ Membership.mem (balancedCore 𝕜 s) ((fun x => HSMul.hSMul a x) y)
  -/
  rw [mem_balancedCore_iff] at hy
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹ : SeminormedRing 𝕜
    inst✝ : SMul 𝕜 E
    s : Set E
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    y : E
    hy : Exists fun t => And (Balanced 𝕜 t) (And (HasSubset.Subset t s) (Membershi …
    ⊢ Membership.mem (balancedCore 𝕜 s) ((fun x => HSMul.hSMul a x) y)
  -/
  rcases hy with ⟨t, ht1, ht2, hy⟩
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹ : SeminormedRing 𝕜
    inst✝ : SMul 𝕜 E
    s : Set E
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    y : E
    t : Set E
    ht1 : Balanced 𝕜 t
    ht2 : HasSubset.Subset t s
    hy : Membership.mem t y
    ⊢ Membership.mem (balancedCore 𝕜 s) ((fun x => HSMul.hSMul a x) y)
  -/
  exact ⟨t, ⟨ht1, ht2⟩, ht1 a ha (smul_mem_smul_set hy)⟩
  /-
    🎉 no goals
  -/


theorem balancedCore_balanced (s : Set E) : Balanced 𝕜 (balancedCore 𝕜 s) := fun _ =>
  smul_balancedCore_subset s


/-- The balanced core of `t` is maximal in the sense that it contains any balanced subset
`s` of `t`. -/
theorem Balanced.subset_balancedCore_of_subset (hs : Balanced 𝕜 s) (h : s ⊆ t) :
    s ⊆ balancedCore 𝕜 t :=
  subset_sUnion_of_mem ⟨hs, h⟩


theorem mem_balancedCoreAux_iff : x ∈ balancedCoreAux 𝕜 s ↔ ∀ r : 𝕜, 1 ≤ ‖r‖ → x ∈ r • s :=
  mem_iInter₂


theorem mem_balancedHull_iff : x ∈ balancedHull 𝕜 s ↔ ∃ r : 𝕜, ‖r‖ ≤ 1 ∧ x ∈ r • s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹ : SeminormedRing 𝕜
    inst✝ : SMul 𝕜 E
    s : Set E
    x : E
    ⊢ Iff (Membership.mem (balancedHull 𝕜 s) x) (Exists fun r => And (LE.le (Norm. …
  -/
  simp [balancedHull]
  /-
    🎉 no goals
  -/


/-- The balanced hull of `s` is minimal in the sense that it is contained in any balanced superset
`t` of `s`. -/
theorem Balanced.balancedHull_subset_of_subset (ht : Balanced 𝕜 t) (h : s ⊆ t) :
    balancedHull 𝕜 s ⊆ t := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹ : SeminormedRing 𝕜
    inst✝ : SMul 𝕜 E
    s t : Set E
    ht : Balanced 𝕜 t
    h : HasSubset.Subset s t
    ⊢ HasSubset.Subset (balancedHull 𝕜 s) t
  -/
  intros x hx
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹ : SeminormedRing 𝕜
    inst✝ : SMul 𝕜 E
    s t : Set E
    ht : Balanced 𝕜 t
    h : HasSubset.Subset s t
    x : E
    hx : Membership.mem (balancedHull 𝕜 s) x
    ⊢ Membership.mem t x
  -/
  obtain ⟨r, hr, y, hy, rfl⟩ := mem_balancedHull_iff.1 hx
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹ : SeminormedRing 𝕜
    inst✝ : SMul 𝕜 E
    s t : Set E
    ht : Balanced 𝕜 t
    h : HasSubset.Subset s t
    r : 𝕜
    hr : LE.le (Norm.norm r) 1
    y : E
    hy : Membership.mem s y
    hx : Membership.mem (balancedHull 𝕜 s) ((fun x => HSMul.hSMul r x) y)
    ⊢ Membership.mem t ((fun x => HSMul.hSMul r x) y)
  -/
  exact ht.smul_mem hr (h hy)
  /-
    🎉 no goals
  -/


@[mono, gcongr]
theorem balancedHull_mono (hst : s ⊆ t) : balancedHull 𝕜 s ⊆ balancedHull 𝕜 t := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹ : SeminormedRing 𝕜
    inst✝ : SMul 𝕜 E
    s t : Set E
    hst : HasSubset.Subset s t
    ⊢ HasSubset.Subset (balancedHull 𝕜 s) (balancedHull 𝕜 t)
  -/
  intro x hx
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹ : SeminormedRing 𝕜
    inst✝ : SMul 𝕜 E
    s t : Set E
    hst : HasSubset.Subset s t
    x : E
    hx : Membership.mem (balancedHull 𝕜 s) x
    ⊢ Membership.mem (balancedHull 𝕜 t) x
  -/
  rw [mem_balancedHull_iff] at *
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹ : SeminormedRing 𝕜
    inst✝ : SMul 𝕜 E
    s t : Set E
    hst : HasSubset.Subset s t
    x : E
    hx : Exists fun r => And (LE.le (Norm.norm r) 1) (Membership.mem (HSMul.hSMul  …
    ⊢ Exists fun r => And (LE.le (Norm.norm r) 1) (Membership.mem (HSMul.hSMul r t …
  -/
  obtain ⟨r, hr₁, hr₂⟩ := hx
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹ : SeminormedRing 𝕜
    inst✝ : SMul 𝕜 E
    s t : Set E
    hst : HasSubset.Subset s t
    x : E
    r : 𝕜
    hr₁ : LE.le (Norm.norm r) 1
    hr₂ : Membership.mem (HSMul.hSMul r s) x
    ⊢ Exists fun r => And (LE.le (Norm.norm r) 1) (Membership.mem (HSMul.hSMul r t …
  -/
  use r
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹ : SeminormedRing 𝕜
    inst✝ : SMul 𝕜 E
    s t : Set E
    hst : HasSubset.Subset s t
    x : E
    r : 𝕜
    hr₁ : LE.le (Norm.norm r) 1
    hr₂ : Membership.mem (HSMul.hSMul r s) x
    ⊢ And (LE.le (Norm.norm r) 1) (Membership.mem (HSMul.hSMul r t) x)
  -/
  exact ⟨hr₁, smul_set_mono hst hr₂⟩
  /-
    🎉 no goals
  -/


theorem balancedCore_zero_mem (hs : (0 : E) ∈ s) : (0 : E) ∈ balancedCore 𝕜 s :=
  mem_balancedCore_iff.2 ⟨0, balanced_zero, zero_subset.2 hs, Set.zero_mem_zero⟩


theorem balancedCore_nonempty_iff : (balancedCore 𝕜 s).Nonempty ↔ (0 : E) ∈ s :=
  ⟨fun h => zero_subset.1 <| (zero_smul_set h).superset.trans <|
    (balancedCore_balanced s (0 : 𝕜) <| norm_zero.trans_le zero_le_one).trans <|
      balancedCore_subset _,
    fun h => ⟨0, balancedCore_zero_mem h⟩⟩


theorem subset_balancedHull [NormOneClass 𝕜] {s : Set E} : s ⊆ balancedHull 𝕜 s := fun _ hx =>
  mem_balancedHull_iff.2 ⟨1, norm_one.le, _, hx, one_smul _ _⟩


theorem balancedHull.balanced (s : Set E) : Balanced 𝕜 (balancedHull 𝕜 s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    ⊢ Balanced 𝕜 (balancedHull 𝕜 s)
  -/
  intro a ha
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    ⊢ HasSubset.Subset (HSMul.hSMul a (balancedHull 𝕜 s)) (balancedHull 𝕜 s)
  -/
  simp_rw [balancedHull, smul_set_iUnion₂, subset_def, mem_iUnion₂]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    ⊢ ∀ (x : E), (Exists fun i => Exists fun j => Membership.mem (HSMul.hSMul a (H …
  -/
  rintro x ⟨r, hr, hx⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    x : E
    r : 𝕜
    hr : LE.le (Norm.norm r) 1
    hx : Membership.mem (HSMul.hSMul a (HSMul.hSMul r s)) x
    ⊢ Exists fun i => Exists fun j => Membership.mem (HSMul.hSMul i s) x
  -/
  rw [← smul_assoc] at hx
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    x : E
    r : 𝕜
    hr : LE.le (Norm.norm r) 1
    hx : Membership.mem (HSMul.hSMul (HSMul.hSMul a r) s) x
    ⊢ Exists fun i => Exists fun j => Membership.mem (HSMul.hSMul i s) x
  -/
  exact ⟨a • r, (SeminormedRing.norm_mul _ _).trans (mul_le_one₀ ha (norm_nonneg r) hr), hx⟩
  /-
    🎉 no goals
  -/


open Balanced in
theorem balancedHull_add_subset [NormOneClass 𝕜] {t : Set E} :
    balancedHull 𝕜 (s + t) ⊆ balancedHull 𝕜 s + balancedHull 𝕜 t :=
  balancedHull_subset_of_subset (add (balancedHull.balanced _) (balancedHull.balanced _))
    (add_subset_add (subset_balancedHull _) (subset_balancedHull _))


@[simp]
theorem balancedCoreAux_empty : balancedCoreAux 𝕜 (∅ : Set E) = ∅ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    ⊢ Eq (balancedCoreAux 𝕜 EmptyCollection.emptyCollection) EmptyCollection.empty …
  -/
  simp_rw [balancedCoreAux, iInter₂_eq_empty_iff, smul_set_empty]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    ⊢ ∀ (a : E), Exists fun i => Exists fun h => Not (Membership.mem EmptyCollecti …
  -/
  exact fun _ => ⟨1, norm_one.ge, not_mem_empty _⟩
  /-
    🎉 no goals
  -/


theorem balancedCoreAux_subset (s : Set E) : balancedCoreAux 𝕜 s ⊆ s := fun x hx => by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    hx : Membership.mem (balancedCoreAux 𝕜 s) x
    ⊢ Membership.mem s x
  -/
  simpa only [one_smul] using mem_balancedCoreAux_iff.1 hx 1 norm_one.ge
  /-
    🎉 no goals
  -/


theorem balancedCoreAux_balanced (h0 : (0 : E) ∈ balancedCoreAux 𝕜 s) :
    Balanced 𝕜 (balancedCoreAux 𝕜 s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    h0 : Membership.mem (balancedCoreAux 𝕜 s) 0
    ⊢ Balanced 𝕜 (balancedCoreAux 𝕜 s)
  -/
  rintro a ha x ⟨y, hy, rfl⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    h0 : Membership.mem (balancedCoreAux 𝕜 s) 0
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    y : E
    hy : Membership.mem (balancedCoreAux 𝕜 s) y
    ⊢ Membership.mem (balancedCoreAux 𝕜 s) ((fun x => HSMul.hSMul a x) y)
  -/
  obtain rfl | h := eq_or_ne a 0
    /-
      case intro.intro.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NormedDivisionRing 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      h0 : Membership.mem (balancedCoreAux 𝕜 s) 0
      y : E
      hy : Membership.mem (balancedCoreAux 𝕜 s) y
      ha : LE.le (Norm.norm 0) 1
      ⊢ Membership.mem (balancedCoreAux 𝕜 s) ((fun x => HSMul.hSMul 0 x) y)
    -/
  · simp_rw [zero_smul, h0]
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.inr
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    h0 : Membership.mem (balancedCoreAux 𝕜 s) 0
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    y : E
    hy : Membership.mem (balancedCoreAux 𝕜 s) y
    h : Ne a 0
    ⊢ Membership.mem (balancedCoreAux 𝕜 s) ((fun x => HSMul.hSMul a x) y)
  -/
  rw [mem_balancedCoreAux_iff] at hy ⊢
  /-
    case intro.intro.inr
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    h0 : Membership.mem (balancedCoreAux 𝕜 s) 0
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    y : E
    hy : ∀ (r : 𝕜), LE.le 1 (Norm.norm r) → Membership.mem (HSMul.hSMul r s) y
    h : Ne a 0
    ⊢ ∀ (r : 𝕜), LE.le 1 (Norm.norm r) → Membership.mem (HSMul.hSMul r s) ((fun x  …
  -/
  intro r hr
  have h'' : 1 ≤ ‖a⁻¹ • r‖ := by
    rw [norm_smul, norm_inv]
    exact one_le_mul_of_one_le_of_one_le ((one_le_inv₀ (norm_pos_iff.mpr h)).2 ha) hr
  /-
    case intro.intro.inr
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    h0 : Membership.mem (balancedCoreAux 𝕜 s) 0
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    y : E
    hy : ∀ (r : 𝕜), LE.le 1 (Norm.norm r) → Membership.mem (HSMul.hSMul r s) y
    h : Ne a 0
    r : 𝕜
    hr : LE.le 1 (Norm.norm r)
    h'' : LE.le 1 (Norm.norm (HSMul.hSMul (Inv.inv a) r))
    ⊢ Membership.mem (HSMul.hSMul r s) ((fun x => HSMul.hSMul a x) y)
  -/
  have h' := hy (a⁻¹ • r) h''
  /-
    case intro.intro.inr
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    h0 : Membership.mem (balancedCoreAux 𝕜 s) 0
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    y : E
    hy : ∀ (r : 𝕜), LE.le 1 (Norm.norm r) → Membership.mem (HSMul.hSMul r s) y
    h : Ne a 0
    r : 𝕜
    hr : LE.le 1 (Norm.norm r)
    h'' : LE.le 1 (Norm.norm (HSMul.hSMul (Inv.inv a) r))
    h' : Membership.mem (HSMul.hSMul (HSMul.hSMul (Inv.inv a) r) s) y
    ⊢ Membership.mem (HSMul.hSMul r s) ((fun x => HSMul.hSMul a x) y)
  -/
  rwa [smul_assoc, mem_inv_smul_set_iff₀ h] at h'
  /-
    🎉 no goals
  -/


theorem balancedCoreAux_maximal (h : t ⊆ s) (ht : Balanced 𝕜 t) : t ⊆ balancedCoreAux 𝕜 s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    h : HasSubset.Subset t s
    ht : Balanced 𝕜 t
    ⊢ HasSubset.Subset t (balancedCoreAux 𝕜 s)
  -/
  refine fun x hx => mem_balancedCoreAux_iff.2 fun r hr => ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    h : HasSubset.Subset t s
    ht : Balanced 𝕜 t
    x : E
    hx : Membership.mem t x
    r : 𝕜
    hr : LE.le 1 (Norm.norm r)
    ⊢ Membership.mem (HSMul.hSMul r s) x
  -/
  rw [mem_smul_set_iff_inv_smul_mem₀ (norm_pos_iff.mp <| zero_lt_one.trans_le hr)]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    h : HasSubset.Subset t s
    ht : Balanced 𝕜 t
    x : E
    hx : Membership.mem t x
    r : 𝕜
    hr : LE.le 1 (Norm.norm r)
    ⊢ Membership.mem s (HSMul.hSMul (Inv.inv r) x)
  -/
  refine h (ht.smul_mem ?_ hx)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    h : HasSubset.Subset t s
    ht : Balanced 𝕜 t
    x : E
    hx : Membership.mem t x
    r : 𝕜
    hr : LE.le 1 (Norm.norm r)
    ⊢ LE.le (Norm.norm (Inv.inv r)) 1
  -/
  rw [norm_inv]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    h : HasSubset.Subset t s
    ht : Balanced 𝕜 t
    x : E
    hx : Membership.mem t x
    r : 𝕜
    hr : LE.le 1 (Norm.norm r)
    ⊢ LE.le (Inv.inv (Norm.norm r)) 1
  -/
  exact inv_le_one_of_one_le₀ hr
  /-
    🎉 no goals
  -/


theorem balancedCore_subset_balancedCoreAux : balancedCore 𝕜 s ⊆ balancedCoreAux 𝕜 s :=
  balancedCoreAux_maximal (balancedCore_subset s) (balancedCore_balanced s)


theorem balancedCore_eq_iInter (hs : (0 : E) ∈ s) :
    balancedCore 𝕜 s = ⋂ (r : 𝕜) (_ : 1 ≤ ‖r‖), r • s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Membership.mem s 0
    ⊢ Eq (balancedCore 𝕜 s) (Set.iInter fun r => Set.iInter fun x => HSMul.hSMul r …
  -/
  refine balancedCore_subset_balancedCoreAux.antisymm ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Membership.mem s 0
    ⊢ HasSubset.Subset (balancedCoreAux 𝕜 s) (balancedCore 𝕜 s)
  -/
  refine (balancedCoreAux_balanced ?_).subset_balancedCore_of_subset (balancedCoreAux_subset s)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Membership.mem s 0
    ⊢ Membership.mem (balancedCoreAux 𝕜 s) 0
  -/
  exact balancedCore_subset_balancedCoreAux (balancedCore_zero_mem hs)
  /-
    🎉 no goals
  -/


theorem subset_balancedCore (ht : (0 : E) ∈ t) (hst : ∀ a : 𝕜, ‖a‖ ≤ 1 → a • s ⊆ t) :
    s ⊆ balancedCore 𝕜 t := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    ht : Membership.mem t 0
    hst : ∀ (a : 𝕜), LE.le (Norm.norm a) 1 → HasSubset.Subset (HSMul.hSMul a s) t
    ⊢ HasSubset.Subset s (balancedCore 𝕜 t)
  -/
  rw [balancedCore_eq_iInter ht]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    ht : Membership.mem t 0
    hst : ∀ (a : 𝕜), LE.le (Norm.norm a) 1 → HasSubset.Subset (HSMul.hSMul a s) t
    ⊢ HasSubset.Subset s (Set.iInter fun r => Set.iInter fun x => HSMul.hSMul r t)
  -/
  refine subset_iInter₂ fun a ha ↦ ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    ht : Membership.mem t 0
    hst : ∀ (a : 𝕜), LE.le (Norm.norm a) 1 → HasSubset.Subset (HSMul.hSMul a s) t
    a : 𝕜
    ha : LE.le 1 (Norm.norm a)
    ⊢ HasSubset.Subset s (HSMul.hSMul a t)
  -/
  rw [subset_set_smul_iff₀ (norm_pos_iff.mp <| zero_lt_one.trans_le ha)]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    ht : Membership.mem t 0
    hst : ∀ (a : 𝕜), LE.le (Norm.norm a) 1 → HasSubset.Subset (HSMul.hSMul a s) t
    a : 𝕜
    ha : LE.le 1 (Norm.norm a)
    ⊢ HasSubset.Subset (HSMul.hSMul (Inv.inv a) s) t
  -/
  apply hst
  /-
    case a
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    ht : Membership.mem t 0
    hst : ∀ (a : 𝕜), LE.le (Norm.norm a) 1 → HasSubset.Subset (HSMul.hSMul a s) t
    a : 𝕜
    ha : LE.le 1 (Norm.norm a)
    ⊢ LE.le (Norm.norm (Inv.inv a)) 1
  -/
  rw [norm_inv]
  /-
    case a
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    ht : Membership.mem t 0
    hst : ∀ (a : 𝕜), LE.le (Norm.norm a) 1 → HasSubset.Subset (HSMul.hSMul a s) t
    a : 𝕜
    ha : LE.le 1 (Norm.norm a)
    ⊢ LE.le (Inv.inv (Norm.norm a)) 1
  -/
  exact inv_le_one_of_one_le₀ ha
  /-
    🎉 no goals
  -/


protected theorem IsClosed.balancedCore (hU : IsClosed U) : IsClosed (balancedCore 𝕜 U) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : NormedDivisionRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul 𝕜 E
    U : Set E
    hU : IsClosed U
    ⊢ IsClosed (balancedCore 𝕜 U)
  -/
  by_cases h : (0 : E) ∈ U
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : NormedDivisionRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : TopologicalSpace E
      inst✝ : ContinuousSMul 𝕜 E
      U : Set E
      hU : IsClosed U
      h : Membership.mem U 0
      ⊢ IsClosed (balancedCore 𝕜 U)
    -/
  · rw [balancedCore_eq_iInter h]
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : NormedDivisionRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : TopologicalSpace E
      inst✝ : ContinuousSMul 𝕜 E
      U : Set E
      hU : IsClosed U
      h : Membership.mem U 0
      ⊢ IsClosed (Set.iInter fun r => Set.iInter fun x => HSMul.hSMul r U)
    -/
    refine isClosed_iInter fun a => ?_
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : NormedDivisionRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : TopologicalSpace E
      inst✝ : ContinuousSMul 𝕜 E
      U : Set E
      hU : IsClosed U
      h : Membership.mem U 0
      a : 𝕜
      ⊢ IsClosed (Set.iInter fun x => HSMul.hSMul a U)
    -/
    refine isClosed_iInter fun ha => ?_
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : NormedDivisionRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : TopologicalSpace E
      inst✝ : ContinuousSMul 𝕜 E
      U : Set E
      hU : IsClosed U
      h : Membership.mem U 0
      a : 𝕜
      ha : LE.le 1 (Norm.norm a)
      ⊢ IsClosed (HSMul.hSMul a U)
    -/
    have ha' := lt_of_lt_of_le zero_lt_one ha
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : NormedDivisionRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : TopologicalSpace E
      inst✝ : ContinuousSMul 𝕜 E
      U : Set E
      hU : IsClosed U
      h : Membership.mem U 0
      a : 𝕜
      ha : LE.le 1 (Norm.norm a)
      ha' : LT.lt 0 (Norm.norm a)
      ⊢ IsClosed (HSMul.hSMul a U)
    -/
    rw [norm_pos_iff] at ha'
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : NormedDivisionRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : TopologicalSpace E
      inst✝ : ContinuousSMul 𝕜 E
      U : Set E
      hU : IsClosed U
      h : Membership.mem U 0
      a : 𝕜
      ha : LE.le 1 (Norm.norm a)
      ha' : Ne a 0
      ⊢ IsClosed (HSMul.hSMul a U)
    -/
    exact isClosedMap_smul_of_ne_zero ha' U hU
    /-
      🎉 no goals
    -/
  · have : balancedCore 𝕜 U = ∅ := by
      contrapose! h
      exact balancedCore_nonempty_iff.mp h
    /-
      case neg
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : NormedDivisionRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : TopologicalSpace E
      inst✝ : ContinuousSMul 𝕜 E
      U : Set E
      hU : IsClosed U
      h : Not (Membership.mem U 0)
      this : Eq (balancedCore 𝕜 U) EmptyCollection.emptyCollection
      ⊢ IsClosed (balancedCore 𝕜 U)
    -/
    rw [this]
    /-
      case neg
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : NormedDivisionRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : TopologicalSpace E
      inst✝ : ContinuousSMul 𝕜 E
      U : Set E
      hU : IsClosed U
      h : Not (Membership.mem U 0)
      this : Eq (balancedCore 𝕜 U) EmptyCollection.emptyCollection
      ⊢ IsClosed EmptyCollection.emptyCollection
    -/
    exact isClosed_empty
    /-
      🎉 no goals
    -/

-- We don't have a `NontriviallyNormedDivisionRing`, so we use a `NeBot` assumption instead

theorem balancedCore_mem_nhds_zero (hU : U ∈ 𝓝 (0 : E)) : balancedCore 𝕜 U ∈ 𝓝 (0 : E) := by
  -- Getting neighborhoods of the origin for `0 : 𝕜` and `0 : E`
  obtain ⟨r, V, hr, hV, hrVU⟩ : ∃ (r : ℝ) (V : Set E),
      0 < r ∧ V ∈ 𝓝 (0 : E) ∧ ∀ (c : 𝕜) (y : E), ‖c‖ < r → y ∈ V → c • y ∈ U := by
    have h : Filter.Tendsto (fun x : 𝕜 × E => x.fst • x.snd) (𝓝 (0, 0)) (𝓝 0) :=
      continuous_smul.tendsto' (0, 0) _ (smul_zero _)
    simpa only [← Prod.exists', ← Prod.forall', ← and_imp, ← and_assoc, exists_prop] using
      h.basis_left (NormedAddCommGroup.nhds_zero_basis_norm_lt.prod_nhds (𝓝 _).basis_sets) U hU
  obtain ⟨y, hyr, hy₀⟩ : ∃ y : 𝕜, ‖y‖ < r ∧ y ≠ 0 :=
    Filter.nonempty_of_mem <|
      (nhdsWithin_hasBasis NormedAddCommGroup.nhds_zero_basis_norm_lt {0}ᶜ).mem_of_mem hr
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : NormedDivisionRing 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    U : Set E
    inst✝ : (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0))).NeBot
    hU : Membership.mem (nhds 0) U
    r : Real
    V : Set E
    hr : LT.lt 0 r
    hV : Membership.mem (nhds 0) V
    hrVU : ∀ (c : 𝕜) (y : E), LT.lt (Norm.norm c) r → Membership.mem V y → Members …
    y : 𝕜
    hyr : LT.lt (Norm.norm y) r
    hy₀ : Ne y 0
    ⊢ Membership.mem (nhds 0) (balancedCore 𝕜 U)
  -/
  have : y • V ∈ 𝓝 (0 : E) := (set_smul_mem_nhds_zero_iff hy₀).mpr hV
  -- It remains to show that `y • V ⊆ balancedCore 𝕜 U`
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : NormedDivisionRing 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    U : Set E
    inst✝ : (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0))).NeBot
    hU : Membership.mem (nhds 0) U
    r : Real
    V : Set E
    hr : LT.lt 0 r
    hV : Membership.mem (nhds 0) V
    hrVU : ∀ (c : 𝕜) (y : E), LT.lt (Norm.norm c) r → Membership.mem V y → Members …
    y : 𝕜
    hyr : LT.lt (Norm.norm y) r
    hy₀ : Ne y 0
    this : Membership.mem (nhds 0) (HSMul.hSMul y V)
    ⊢ Membership.mem (nhds 0) (balancedCore 𝕜 U)
  -/
  refine Filter.mem_of_superset this (subset_balancedCore (mem_of_mem_nhds hU) fun a ha => ?_)
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : NormedDivisionRing 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    U : Set E
    inst✝ : (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0))).NeBot
    hU : Membership.mem (nhds 0) U
    r : Real
    V : Set E
    hr : LT.lt 0 r
    hV : Membership.mem (nhds 0) V
    hrVU : ∀ (c : 𝕜) (y : E), LT.lt (Norm.norm c) r → Membership.mem V y → Members …
    y : 𝕜
    hyr : LT.lt (Norm.norm y) r
    hy₀ : Ne y 0
    this : Membership.mem (nhds 0) (HSMul.hSMul y V)
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    ⊢ HasSubset.Subset (HSMul.hSMul a (HSMul.hSMul y V)) U
  -/
  rw [smul_smul]
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : NormedDivisionRing 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    U : Set E
    inst✝ : (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0))).NeBot
    hU : Membership.mem (nhds 0) U
    r : Real
    V : Set E
    hr : LT.lt 0 r
    hV : Membership.mem (nhds 0) V
    hrVU : ∀ (c : 𝕜) (y : E), LT.lt (Norm.norm c) r → Membership.mem V y → Members …
    y : 𝕜
    hyr : LT.lt (Norm.norm y) r
    hy₀ : Ne y 0
    this : Membership.mem (nhds 0) (HSMul.hSMul y V)
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    ⊢ HasSubset.Subset (HSMul.hSMul (HMul.hMul a y) V) U
  -/
  rintro _ ⟨z, hz, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : NormedDivisionRing 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    U : Set E
    inst✝ : (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0))).NeBot
    hU : Membership.mem (nhds 0) U
    r : Real
    V : Set E
    hr : LT.lt 0 r
    hV : Membership.mem (nhds 0) V
    hrVU : ∀ (c : 𝕜) (y : E), LT.lt (Norm.norm c) r → Membership.mem V y → Members …
    y : 𝕜
    hyr : LT.lt (Norm.norm y) r
    hy₀ : Ne y 0
    this : Membership.mem (nhds 0) (HSMul.hSMul y V)
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    z : E
    hz : Membership.mem V z
    ⊢ Membership.mem U ((fun x => HSMul.hSMul (HMul.hMul a y) x) z)
  -/
  refine hrVU _ _ ?_ hz
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : NormedDivisionRing 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    U : Set E
    inst✝ : (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0))).NeBot
    hU : Membership.mem (nhds 0) U
    r : Real
    V : Set E
    hr : LT.lt 0 r
    hV : Membership.mem (nhds 0) V
    hrVU : ∀ (c : 𝕜) (y : E), LT.lt (Norm.norm c) r → Membership.mem V y → Members …
    y : 𝕜
    hyr : LT.lt (Norm.norm y) r
    hy₀ : Ne y 0
    this : Membership.mem (nhds 0) (HSMul.hSMul y V)
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    z : E
    hz : Membership.mem V z
    ⊢ LT.lt (Norm.norm (HMul.hMul a y)) r
  -/
  rw [norm_mul, ← one_mul r]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : NormedDivisionRing 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    U : Set E
    inst✝ : (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0))).NeBot
    hU : Membership.mem (nhds 0) U
    r : Real
    V : Set E
    hr : LT.lt 0 r
    hV : Membership.mem (nhds 0) V
    hrVU : ∀ (c : 𝕜) (y : E), LT.lt (Norm.norm c) r → Membership.mem V y → Members …
    y : 𝕜
    hyr : LT.lt (Norm.norm y) r
    hy₀ : Ne y 0
    this : Membership.mem (nhds 0) (HSMul.hSMul y V)
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    z : E
    hz : Membership.mem V z
    ⊢ LT.lt (HMul.hMul (Norm.norm a) (Norm.norm y)) (HMul.hMul 1 r)
  -/
  exact mul_lt_mul' ha hyr (norm_nonneg y) one_pos
  /-
    🎉 no goals
  -/


theorem nhds_basis_balanced :
    (𝓝 (0 : E)).HasBasis (fun s : Set E => s ∈ 𝓝 (0 : E) ∧ Balanced 𝕜 s) id :=
  Filter.hasBasis_self.mpr fun s hs =>
    ⟨balancedCore 𝕜 s, balancedCore_mem_nhds_zero hs, balancedCore_balanced s,
      balancedCore_subset s⟩


theorem nhds_basis_closed_balanced [RegularSpace E] :
    (𝓝 (0 : E)).HasBasis (fun s : Set E => s ∈ 𝓝 (0 : E) ∧ IsClosed s ∧ Balanced 𝕜 s) id := by
  refine
    (closed_nhds_basis 0).to_hasBasis (fun s hs => ?_) fun s hs => ⟨s, ⟨hs.1, hs.2.1⟩, rfl.subset⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : NormedDivisionRing 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : TopologicalSpace E
    inst✝² : ContinuousSMul 𝕜 E
    inst✝¹ : (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0))).NeBot
    inst✝ : RegularSpace E
    s : Set E
    hs : And (Membership.mem (nhds 0) s) (IsClosed s)
    ⊢ Exists fun i' => And (And (Membership.mem (nhds 0) i') (And (IsClosed i') (B …
  -/
  refine ⟨balancedCore 𝕜 s, ⟨balancedCore_mem_nhds_zero hs.1, ?_⟩, balancedCore_subset s⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : NormedDivisionRing 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : TopologicalSpace E
    inst✝² : ContinuousSMul 𝕜 E
    inst✝¹ : (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0))).NeBot
    inst✝ : RegularSpace E
    s : Set E
    hs : And (Membership.mem (nhds 0) s) (IsClosed s)
    ⊢ And (IsClosed (balancedCore 𝕜 s)) (Balanced 𝕜 (balancedCore 𝕜 s))
  -/
  exact ⟨hs.2.balancedCore, balancedCore_balanced s⟩
  /-
    🎉 no goals
  -/


