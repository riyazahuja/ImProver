/-- Evaluation at a point is an `AddMonoidHom`. This is the finitely-supported version of
`Pi.evalAddMonoidHom`. -/
def evalAddMonoidHom [∀ i, AddZeroClass (β i)] (i : ι) : (Π₀ i, β i) →+ β i :=
  (Pi.evalAddMonoidHom β i).comp coeFnAddMonoidHom


@[simp, norm_cast]
theorem coe_finset_sum {α} [∀ i, AddCommMonoid (β i)] (s : Finset α) (g : α → Π₀ i, β i) :
    ⇑(∑ a ∈ s, g a) = ∑ a ∈ s, ⇑(g a) :=
  map_sum coeFnAddMonoidHom g s


@[simp]
theorem finset_sum_apply {α} [∀ i, AddCommMonoid (β i)] (s : Finset α) (g : α → Π₀ i, β i) (i : ι) :
    (∑ a ∈ s, g a) i = ∑ a ∈ s, g a i :=
  map_sum (evalAddMonoidHom i) g s


/-- `DFinsupp.prod f g` is the product of `g i (f i)` over the support of `f`. -/
@[to_additive "`sum f g` is the sum of `g i (f i)` over the support of `f`."]
def prod [∀ i, Zero (β i)] [∀ (i) (x : β i), Decidable (x ≠ 0)] [CommMonoid γ] (f : Π₀ i, β i)
    (g : ∀ i, β i → γ) : γ :=
  ∏ i ∈ f.support, g i (f i)


@[to_additive (attr := simp)]
theorem _root_.map_dfinsupp_prod
    {R S H : Type*} [∀ i, Zero (β i)] [∀ (i) (x : β i), Decidable (x ≠ 0)]
    [CommMonoid R] [CommMonoid S] [FunLike H R S] [MonoidHomClass H R S] (h : H) (f : Π₀ i, β i)
    (g : ∀ i, β i → R) : h (f.prod g) = f.prod fun a b => h (g a b) :=
  map_prod _ _ _


@[to_additive]
theorem prod_mapRange_index {β₁ : ι → Type v₁} {β₂ : ι → Type v₂} [∀ i, Zero (β₁ i)]
    [∀ i, Zero (β₂ i)] [∀ (i) (x : β₁ i), Decidable (x ≠ 0)] [∀ (i) (x : β₂ i), Decidable (x ≠ 0)]
    [CommMonoid γ] {f : ∀ i, β₁ i → β₂ i} {hf : ∀ i, f i 0 = 0} {g : Π₀ i, β₁ i} {h : ∀ i, β₂ i → γ}
    (h0 : ∀ i, h i 0 = 1) : (mapRange f hf g).prod h = g.prod fun i b => h i (f i b) := by
  /-
    ι : Type u
    γ : Type w
    inst✝⁵ : DecidableEq ι
    β₁ : ι → Type v₁
    β₂ : ι → Type v₂
    inst✝⁴ : (i : ι) → Zero (β₁ i)
    inst✝³ : (i : ι) → Zero (β₂ i)
    inst✝² : (i : ι) → (x : β₁ i) → Decidable (Ne x 0)
    inst✝¹ : (i : ι) → (x : β₂ i) → Decidable (Ne x 0)
    inst✝ : CommMonoid γ
    f : (i : ι) → β₁ i → β₂ i
    hf : ∀ (i : ι), Eq (f i 0) 0
    g : DFinsupp fun i => β₁ i
    h : (i : ι) → β₂ i → γ
    h0 : ∀ (i : ι), Eq (h i 0) 1
    ⊢ Eq ((DFinsupp.mapRange f hf g).prod h) (g.prod fun i b => h i (f i b))
  -/
  rw [mapRange_def]
  /-
    ι : Type u
    γ : Type w
    inst✝⁵ : DecidableEq ι
    β₁ : ι → Type v₁
    β₂ : ι → Type v₂
    inst✝⁴ : (i : ι) → Zero (β₁ i)
    inst✝³ : (i : ι) → Zero (β₂ i)
    inst✝² : (i : ι) → (x : β₁ i) → Decidable (Ne x 0)
    inst✝¹ : (i : ι) → (x : β₂ i) → Decidable (Ne x 0)
    inst✝ : CommMonoid γ
    f : (i : ι) → β₁ i → β₂ i
    hf : ∀ (i : ι), Eq (f i 0) 0
    g : DFinsupp fun i => β₁ i
    h : (i : ι) → β₂ i → γ
    h0 : ∀ (i : ι), Eq (h i 0) 1
    ⊢ Eq ((DFinsupp.mk g.support fun i => f (↑i) (g ↑i)).prod h) (g.prod fun i b = …
  -/
  refine (Finset.prod_subset support_mk_subset ?_).trans ?_
    /-
      case refine_1
      ι : Type u
      γ : Type w
      inst✝⁵ : DecidableEq ι
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝⁴ : (i : ι) → Zero (β₁ i)
      inst✝³ : (i : ι) → Zero (β₂ i)
      inst✝² : (i : ι) → (x : β₁ i) → Decidable (Ne x 0)
      inst✝¹ : (i : ι) → (x : β₂ i) → Decidable (Ne x 0)
      inst✝ : CommMonoid γ
      f : (i : ι) → β₁ i → β₂ i
      hf : ∀ (i : ι), Eq (f i 0) 0
      g : DFinsupp fun i => β₁ i
      h : (i : ι) → β₂ i → γ
      h0 : ∀ (i : ι), Eq (h i 0) 1
      ⊢ ∀ (x : ι), Membership.mem g.support x → Not (Membership.mem (DFinsupp.mk g.s …
    -/
  · intro i h1 h2
    /-
      case refine_1
      ι : Type u
      γ : Type w
      inst✝⁵ : DecidableEq ι
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝⁴ : (i : ι) → Zero (β₁ i)
      inst✝³ : (i : ι) → Zero (β₂ i)
      inst✝² : (i : ι) → (x : β₁ i) → Decidable (Ne x 0)
      inst✝¹ : (i : ι) → (x : β₂ i) → Decidable (Ne x 0)
      inst✝ : CommMonoid γ
      f : (i : ι) → β₁ i → β₂ i
      hf : ∀ (i : ι), Eq (f i 0) 0
      g : DFinsupp fun i => β₁ i
      h : (i : ι) → β₂ i → γ
      h0 : ∀ (i : ι), Eq (h i 0) 1
      i : ι
      h1 : Membership.mem g.support i
      h2 : Not (Membership.mem (DFinsupp.mk g.support fun i => f (↑i) (g ↑i)).suppor …
      ⊢ Eq (h i ((DFinsupp.mk g.support fun i => f (↑i) (g ↑i)) i)) 1
    -/
    simp only [mem_support_toFun, ne_eq] at h1
    simp only [Finset.coe_sort_coe, mem_support_toFun, mk_apply, ne_eq, h1, not_false_iff,
      dite_eq_ite, ite_true, not_not] at h2
    /-
      case refine_1
      ι : Type u
      γ : Type w
      inst✝⁵ : DecidableEq ι
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝⁴ : (i : ι) → Zero (β₁ i)
      inst✝³ : (i : ι) → Zero (β₂ i)
      inst✝² : (i : ι) → (x : β₁ i) → Decidable (Ne x 0)
      inst✝¹ : (i : ι) → (x : β₂ i) → Decidable (Ne x 0)
      inst✝ : CommMonoid γ
      f : (i : ι) → β₁ i → β₂ i
      hf : ∀ (i : ι), Eq (f i 0) 0
      g : DFinsupp fun i => β₁ i
      h : (i : ι) → β₂ i → γ
      h0 : ∀ (i : ι), Eq (h i 0) 1
      i : ι
      h1 : Not (Eq (g i) 0)
      h2 : Eq (f i (g i)) 0
      ⊢ Eq (h i ((DFinsupp.mk g.support fun i => f (↑i) (g ↑i)) i)) 1
    -/
    simp [h2, h0]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u
      γ : Type w
      inst✝⁵ : DecidableEq ι
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝⁴ : (i : ι) → Zero (β₁ i)
      inst✝³ : (i : ι) → Zero (β₂ i)
      inst✝² : (i : ι) → (x : β₁ i) → Decidable (Ne x 0)
      inst✝¹ : (i : ι) → (x : β₂ i) → Decidable (Ne x 0)
      inst✝ : CommMonoid γ
      f : (i : ι) → β₁ i → β₂ i
      hf : ∀ (i : ι), Eq (f i 0) 0
      g : DFinsupp fun i => β₁ i
      h : (i : ι) → β₂ i → γ
      h0 : ∀ (i : ι), Eq (h i 0) 1
      ⊢ Eq (g.support.prod fun x => h x ((DFinsupp.mk g.support fun i => f (↑i) (g ↑ …
    -/
  · refine Finset.prod_congr rfl ?_
    /-
      case refine_2
      ι : Type u
      γ : Type w
      inst✝⁵ : DecidableEq ι
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝⁴ : (i : ι) → Zero (β₁ i)
      inst✝³ : (i : ι) → Zero (β₂ i)
      inst✝² : (i : ι) → (x : β₁ i) → Decidable (Ne x 0)
      inst✝¹ : (i : ι) → (x : β₂ i) → Decidable (Ne x 0)
      inst✝ : CommMonoid γ
      f : (i : ι) → β₁ i → β₂ i
      hf : ∀ (i : ι), Eq (f i 0) 0
      g : DFinsupp fun i => β₁ i
      h : (i : ι) → β₂ i → γ
      h0 : ∀ (i : ι), Eq (h i 0) 1
      ⊢ ∀ (x : ι), Membership.mem g.support x → Eq (h x ((DFinsupp.mk g.support fun  …
    -/
    intro i h1
    /-
      case refine_2
      ι : Type u
      γ : Type w
      inst✝⁵ : DecidableEq ι
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝⁴ : (i : ι) → Zero (β₁ i)
      inst✝³ : (i : ι) → Zero (β₂ i)
      inst✝² : (i : ι) → (x : β₁ i) → Decidable (Ne x 0)
      inst✝¹ : (i : ι) → (x : β₂ i) → Decidable (Ne x 0)
      inst✝ : CommMonoid γ
      f : (i : ι) → β₁ i → β₂ i
      hf : ∀ (i : ι), Eq (f i 0) 0
      g : DFinsupp fun i => β₁ i
      h : (i : ι) → β₂ i → γ
      h0 : ∀ (i : ι), Eq (h i 0) 1
      i : ι
      h1 : Membership.mem g.support i
      ⊢ Eq (h i ((DFinsupp.mk g.support fun i => f (↑i) (g ↑i)) i)) ((fun i b => h i …
    -/
    simp only [mem_support_toFun, ne_eq] at h1
    /-
      case refine_2
      ι : Type u
      γ : Type w
      inst✝⁵ : DecidableEq ι
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝⁴ : (i : ι) → Zero (β₁ i)
      inst✝³ : (i : ι) → Zero (β₂ i)
      inst✝² : (i : ι) → (x : β₁ i) → Decidable (Ne x 0)
      inst✝¹ : (i : ι) → (x : β₂ i) → Decidable (Ne x 0)
      inst✝ : CommMonoid γ
      f : (i : ι) → β₁ i → β₂ i
      hf : ∀ (i : ι), Eq (f i 0) 0
      g : DFinsupp fun i => β₁ i
      h : (i : ι) → β₂ i → γ
      h0 : ∀ (i : ι), Eq (h i 0) 1
      i : ι
      h1 : Not (Eq (g i) 0)
      ⊢ Eq (h i ((DFinsupp.mk g.support fun i => f (↑i) (g ↑i)) i)) ((fun i b => h i …
    -/
    simp [h1]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem prod_zero_index [∀ i, AddCommMonoid (β i)] [∀ (i) (x : β i), Decidable (x ≠ 0)]
    [CommMonoid γ] {h : ∀ i, β i → γ} : (0 : Π₀ i, β i).prod h = 1 :=
  rfl


@[to_additive]
theorem prod_single_index [∀ i, Zero (β i)] [∀ (i) (x : β i), Decidable (x ≠ 0)] [CommMonoid γ]
    {i : ι} {b : β i} {h : ∀ i, β i → γ} (h_zero : h i 0 = 1) : (single i b).prod h = h i b := by
  /-
    ι : Type u
    γ : Type w
    β : ι → Type v
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    inst✝ : CommMonoid γ
    i : ι
    b : β i
    h : (i : ι) → β i → γ
    h_zero : Eq (h i 0) 1
    ⊢ Eq ((DFinsupp.single i b).prod h) (h i b)
  -/
  by_cases h : b ≠ 0
    /-
      case pos
      ι : Type u
      γ : Type w
      β : ι → Type v
      inst✝³ : DecidableEq ι
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      inst✝ : CommMonoid γ
      i : ι
      b : β i
      h✝ : (i : ι) → β i → γ
      h_zero : Eq (h✝ i 0) 1
      h : Ne b 0
      ⊢ Eq ((DFinsupp.single i b).prod h✝) (h✝ i b)
    -/
  · simp [DFinsupp.prod, support_single_ne_zero h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u
      γ : Type w
      β : ι → Type v
      inst✝³ : DecidableEq ι
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      inst✝ : CommMonoid γ
      i : ι
      b : β i
      h✝ : (i : ι) → β i → γ
      h_zero : Eq (h✝ i 0) 1
      h : Not (Ne b 0)
      ⊢ Eq ((DFinsupp.single i b).prod h✝) (h✝ i b)
    -/
  · rw [not_not] at h
    /-
      case neg
      ι : Type u
      γ : Type w
      β : ι → Type v
      inst✝³ : DecidableEq ι
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      inst✝ : CommMonoid γ
      i : ι
      b : β i
      h✝ : (i : ι) → β i → γ
      h_zero : Eq (h✝ i 0) 1
      h : Eq b 0
      ⊢ Eq ((DFinsupp.single i b).prod h✝) (h✝ i b)
    -/
    simp [h, prod_zero_index, h_zero]
    /-
      case neg
      ι : Type u
      γ : Type w
      β : ι → Type v
      inst✝³ : DecidableEq ι
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      inst✝ : CommMonoid γ
      i : ι
      b : β i
      h✝ : (i : ι) → β i → γ
      h_zero : Eq (h✝ i 0) 1
      h : Eq b 0
      ⊢ Eq (DFinsupp.prod 0 h✝) 1
    -/
    rfl
    /-
      🎉 no goals
    -/


@[to_additive]
theorem prod_neg_index [∀ i, AddGroup (β i)] [∀ (i) (x : β i), Decidable (x ≠ 0)] [CommMonoid γ]
    {g : Π₀ i, β i} {h : ∀ i, β i → γ} (h0 : ∀ i, h i 0 = 1) :
    (-g).prod h = g.prod fun i b => h i (-b) :=
  prod_mapRange_index h0


@[to_additive]
theorem prod_comm {ι₁ ι₂ : Sort _} {β₁ : ι₁ → Type*} {β₂ : ι₂ → Type*} [DecidableEq ι₁]
    [DecidableEq ι₂] [∀ i, Zero (β₁ i)] [∀ i, Zero (β₂ i)] [∀ (i) (x : β₁ i), Decidable (x ≠ 0)]
    [∀ (i) (x : β₂ i), Decidable (x ≠ 0)] [CommMonoid γ] (f₁ : Π₀ i, β₁ i) (f₂ : Π₀ i, β₂ i)
    (h : ∀ i, β₁ i → ∀ i, β₂ i → γ) :
    (f₁.prod fun i₁ x₁ => f₂.prod fun i₂ x₂ => h i₁ x₁ i₂ x₂) =
      f₂.prod fun i₂ x₂ => f₁.prod fun i₁ x₁ => h i₁ x₁ i₂ x₂ :=
  Finset.prod_comm


@[simp]
theorem sum_apply {ι} {β : ι → Type v} {ι₁ : Type u₁} [DecidableEq ι₁] {β₁ : ι₁ → Type v₁}
    [∀ i₁, Zero (β₁ i₁)] [∀ (i) (x : β₁ i), Decidable (x ≠ 0)] [∀ i, AddCommMonoid (β i)]
    {f : Π₀ i₁, β₁ i₁} {g : ∀ i₁, β₁ i₁ → Π₀ i, β i} {i₂ : ι} :
    (f.sum g) i₂ = f.sum fun i₁ b => g i₁ b i₂ :=
  map_sum (evalAddMonoidHom i₂) _ f.support


theorem support_sum {ι₁ : Type u₁} [DecidableEq ι₁] {β₁ : ι₁ → Type v₁} [∀ i₁, Zero (β₁ i₁)]
    [∀ (i) (x : β₁ i), Decidable (x ≠ 0)] [∀ i, AddCommMonoid (β i)]
    [∀ (i) (x : β i), Decidable (x ≠ 0)] {f : Π₀ i₁, β₁ i₁} {g : ∀ i₁, β₁ i₁ → Π₀ i, β i} :
    (f.sum g).support ⊆ f.support.biUnion fun i => (g i (f i)).support := by
  have :
    ∀ i₁ : ι,
      (f.sum fun (i : ι₁) (b : β₁ i) => (g i b) i₁) ≠ 0 → ∃ i : ι₁, f i ≠ 0 ∧ ¬(g i (f i)) i₁ = 0 :=
    fun i₁ h =>
    let ⟨i, hi, Ne⟩ := Finset.exists_ne_zero_of_sum_ne_zero h
    ⟨i, mem_support_iff.1 hi, Ne⟩
  /-
    ι : Type u
    β : ι → Type v
    inst✝⁵ : DecidableEq ι
    ι₁ : Type u₁
    inst✝⁴ : DecidableEq ι₁
    β₁ : ι₁ → Type v₁
    inst✝³ : (i₁ : ι₁) → Zero (β₁ i₁)
    inst✝² : (i : ι₁) → (x : β₁ i) → Decidable (Ne x 0)
    inst✝¹ : (i : ι) → AddCommMonoid (β i)
    inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    f : DFinsupp fun i₁ => β₁ i₁
    g : (i₁ : ι₁) → β₁ i₁ → DFinsupp fun i => β i
    this : ∀ (i₁ : ι), Ne (f.sum fun i b => (g i b) i₁) 0 → Exists fun i => And (N …
    ⊢ HasSubset.Subset (f.sum g).support (f.support.biUnion fun i => (g i (f i)).s …
  -/
  simpa [Finset.subset_iff, mem_support_iff, Finset.mem_biUnion, sum_apply] using this
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem prod_one [∀ i, AddCommMonoid (β i)] [∀ (i) (x : β i), Decidable (x ≠ 0)] [CommMonoid γ]
    {f : Π₀ i, β i} : (f.prod fun _ _ => (1 : γ)) = 1 :=
  Finset.prod_const_one


@[to_additive (attr := simp)]
theorem prod_mul [∀ i, AddCommMonoid (β i)] [∀ (i) (x : β i), Decidable (x ≠ 0)] [CommMonoid γ]
    {f : Π₀ i, β i} {h₁ h₂ : ∀ i, β i → γ} :
    (f.prod fun i b => h₁ i b * h₂ i b) = f.prod h₁ * f.prod h₂ :=
  Finset.prod_mul_distrib


@[to_additive (attr := simp)]
theorem prod_inv [∀ i, AddCommMonoid (β i)] [∀ (i) (x : β i), Decidable (x ≠ 0)] [CommGroup γ]
    {f : Π₀ i, β i} {h : ∀ i, β i → γ} : (f.prod fun i b => (h i b)⁻¹) = (f.prod h)⁻¹ :=
  (map_prod (invMonoidHom : γ →* γ) _ f.support).symm


@[to_additive]
theorem prod_eq_one [∀ i, Zero (β i)] [∀ (i) (x : β i), Decidable (x ≠ 0)] [CommMonoid γ]
    {f : Π₀ i, β i} {h : ∀ i, β i → γ} (hyp : ∀ i, h i (f i) = 1) : f.prod h = 1 :=
  Finset.prod_eq_one fun i _ => hyp i


theorem smul_sum {α : Type*} [Monoid α] [∀ i, Zero (β i)] [∀ (i) (x : β i), Decidable (x ≠ 0)]
    [AddCommMonoid γ] [DistribMulAction α γ] {f : Π₀ i, β i} {h : ∀ i, β i → γ} {c : α} :
    c • f.sum h = f.sum fun a b => c • h a b :=
  Finset.smul_sum


@[to_additive]
theorem prod_add_index [∀ i, AddCommMonoid (β i)] [∀ (i) (x : β i), Decidable (x ≠ 0)]
    [CommMonoid γ] {f g : Π₀ i, β i} {h : ∀ i, β i → γ} (h_zero : ∀ i, h i 0 = 1)
    (h_add : ∀ i b₁ b₂, h i (b₁ + b₂) = h i b₁ * h i b₂) : (f + g).prod h = f.prod h * g.prod h :=
  have f_eq : (∏ i ∈ f.support ∪ g.support, h i (f i)) = f.prod h :=
    (Finset.prod_subset Finset.subset_union_left <| by
        /-
          ι : Type u
          γ : Type w
          β : ι → Type v
          inst✝³ : DecidableEq ι
          inst✝² : (i : ι) → AddCommMonoid (β i)
          inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
          inst✝ : CommMonoid γ
          f g : DFinsupp fun i => β i
          h : (i : ι) → β i → γ
          h_zero : ∀ (i : ι), Eq (h i 0) 1
          h_add : ∀ (i : ι) (b₁ b₂ : β i), Eq (h i (HAdd.hAdd b₁ b₂)) (HMul.hMul (h i b₁ …
          ⊢ ∀ (x : ι), Membership.mem (Union.union f.support g.support) x → Not (Members …
        -/
        simp +contextual [mem_support_iff, h_zero]).symm
        /-
          🎉 no goals
        -/
  have g_eq : (∏ i ∈ f.support ∪ g.support, h i (g i)) = g.prod h :=
    (Finset.prod_subset Finset.subset_union_right <| by
        /-
          ι : Type u
          γ : Type w
          β : ι → Type v
          inst✝³ : DecidableEq ι
          inst✝² : (i : ι) → AddCommMonoid (β i)
          inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
          inst✝ : CommMonoid γ
          f g : DFinsupp fun i => β i
          h : (i : ι) → β i → γ
          h_zero : ∀ (i : ι), Eq (h i 0) 1
          h_add : ∀ (i : ι) (b₁ b₂ : β i), Eq (h i (HAdd.hAdd b₁ b₂)) (HMul.hMul (h i b₁ …
          f_eq : Eq ((Union.union f.support g.support).prod fun i => h i (f i)) (f.prod h)
          ⊢ ∀ (x : ι), Membership.mem (Union.union f.support g.support) x → Not (Members …
        -/
        simp +contextual [mem_support_iff, h_zero]).symm
        /-
          🎉 no goals
        -/
  calc
    (∏ i ∈ (f + g).support, h i ((f + g) i)) = ∏ i ∈ f.support ∪ g.support, h i ((f + g) i) :=
      Finset.prod_subset support_add <| by
        /-
          ι : Type u
          γ : Type w
          β : ι → Type v
          inst✝³ : DecidableEq ι
          inst✝² : (i : ι) → AddCommMonoid (β i)
          inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
          inst✝ : CommMonoid γ
          f g : DFinsupp fun i => β i
          h : (i : ι) → β i → γ
          h_zero : ∀ (i : ι), Eq (h i 0) 1
          h_add : ∀ (i : ι) (b₁ b₂ : β i), Eq (h i (HAdd.hAdd b₁ b₂)) (HMul.hMul (h i b₁ …
          f_eq : Eq ((Union.union f.support g.support).prod fun i => h i (f i)) (f.prod h)
          g_eq : Eq ((Union.union f.support g.support).prod fun i => h i (g i)) (g.prod h)
          ⊢ ∀ (x : ι), Membership.mem (Union.union f.support g.support) x → Not (Members …
        -/
        simp +contextual [mem_support_iff, h_zero]
        /-
          🎉 no goals
        -/
    _ = (∏ i ∈ f.support ∪ g.support, h i (f i)) * ∏ i ∈ f.support ∪ g.support, h i (g i) := by
      /-
        ι : Type u
        γ : Type w
        β : ι → Type v
        inst✝³ : DecidableEq ι
        inst✝² : (i : ι) → AddCommMonoid (β i)
        inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
        inst✝ : CommMonoid γ
        f g : DFinsupp fun i => β i
        h : (i : ι) → β i → γ
        h_zero : ∀ (i : ι), Eq (h i 0) 1
        h_add : ∀ (i : ι) (b₁ b₂ : β i), Eq (h i (HAdd.hAdd b₁ b₂)) (HMul.hMul (h i b₁ …
        f_eq : Eq ((Union.union f.support g.support).prod fun i => h i (f i)) (f.prod h)
        g_eq : Eq ((Union.union f.support g.support).prod fun i => h i (g i)) (g.prod h)
        ⊢ Eq ((Union.union f.support g.support).prod fun i => h i ((HAdd.hAdd f g) i)) …
      -/
      { simp [h_add, Finset.prod_mul_distrib] }
      /-
        🎉 no goals
      -/
                /-
                  ι : Type u
                  γ : Type w
                  β : ι → Type v
                  inst✝³ : DecidableEq ι
                  inst✝² : (i : ι) → AddCommMonoid (β i)
                  inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
                  inst✝ : CommMonoid γ
                  f g : DFinsupp fun i => β i
                  h : (i : ι) → β i → γ
                  h_zero : ∀ (i : ι), Eq (h i 0) 1
                  h_add : ∀ (i : ι) (b₁ b₂ : β i), Eq (h i (HAdd.hAdd b₁ b₂)) (HMul.hMul (h i b₁ …
                  f_eq : Eq ((Union.union f.support g.support).prod fun i => h i (f i)) (f.prod h)
                  g_eq : Eq ((Union.union f.support g.support).prod fun i => h i (g i)) (g.prod h)
                  ⊢ Eq (HMul.hMul ((Union.union f.support g.support).prod fun i => h i (f i)) (( …
                -/
    _ = _ := by rw [f_eq, g_eq]
                /-
                  🎉 no goals
                -/


@[to_additive (attr := simp)]
theorem prod_eq_prod_fintype [Fintype ι] [∀ i, Zero (β i)] [∀ (i : ι) (x : β i), Decidable (x ≠ 0)]
    -- Porting note: `f` was a typeclass argument
    [CommMonoid γ] (v : Π₀ i, β i) {f : ∀ i, β i → γ} (hf : ∀ i, f i 0 = 1) :
    v.prod f = ∏ i, f i (DFinsupp.equivFunOnFintype v i) := by
  /-
    ι : Type u
    γ : Type w
    β : ι → Type v
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    inst✝ : CommMonoid γ
    v : DFinsupp fun i => β i
    f : (i : ι) → β i → γ
    hf : ∀ (i : ι), Eq (f i 0) 1
    ⊢ Eq (v.prod f) (Finset.univ.prod fun i => f i (DFinsupp.equivFunOnFintype v i))
  -/
  suffices (∏ i ∈ v.support, f i (v i)) = ∏ i, f i (v i) by simp [DFinsupp.prod, this]
  /-
    ι : Type u
    γ : Type w
    β : ι → Type v
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    inst✝ : CommMonoid γ
    v : DFinsupp fun i => β i
    f : (i : ι) → β i → γ
    hf : ∀ (i : ι), Eq (f i 0) 1
    ⊢ Eq (v.support.prod fun i => f i (v i)) (Finset.univ.prod fun i => f i (v i))
  -/
  apply Finset.prod_subset v.support.subset_univ
  /-
    ι : Type u
    γ : Type w
    β : ι → Type v
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    inst✝ : CommMonoid γ
    v : DFinsupp fun i => β i
    f : (i : ι) → β i → γ
    hf : ∀ (i : ι), Eq (f i 0) 1
    ⊢ ∀ (x : ι), Membership.mem Finset.univ x → Not (Membership.mem v.support x) → …
  -/
  intro i _ hi
  /-
    ι : Type u
    γ : Type w
    β : ι → Type v
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    inst✝ : CommMonoid γ
    v : DFinsupp fun i => β i
    f : (i : ι) → β i → γ
    hf : ∀ (i : ι), Eq (f i 0) 1
    i : ι
    a✝ : Membership.mem Finset.univ i
    hi : Not (Membership.mem v.support i)
    ⊢ Eq (f i (v i)) 1
  -/
  rw [mem_support_iff, not_not] at hi
  /-
    ι : Type u
    γ : Type w
    β : ι → Type v
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    inst✝ : CommMonoid γ
    v : DFinsupp fun i => β i
    f : (i : ι) → β i → γ
    hf : ∀ (i : ι), Eq (f i 0) 1
    i : ι
    a✝ : Membership.mem Finset.univ i
    hi : Eq (v i) 0
    ⊢ Eq (f i (v i)) 1
  -/
  rw [hi, hf]
  /-
    🎉 no goals
  -/


@[simp]
lemma prod_eq_zero_iff : f.prod g = 0 ↔ ∃ i ∈ f.support, g i (f i) = 0 := Finset.prod_eq_zero_iff

lemma prod_ne_zero_iff : f.prod g ≠ 0 ↔ ∀ i ∈ f.support, g i (f i) ≠ 0 := Finset.prod_ne_zero_iff


/--
When summing over an `AddMonoidHom`, the decidability assumption is not needed, and the result is
also an `AddMonoidHom`.
-/
def sumAddHom [∀ i, AddZeroClass (β i)] [AddCommMonoid γ] (φ : ∀ i, β i →+ γ) :
    (Π₀ i, β i) →+ γ where
  toFun f :=
    (f.support'.lift fun s => ∑ i ∈ Multiset.toFinset s.1, φ i (f i)) <| by
      /-
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → AddZeroClass (β i)
        inst✝ : AddCommMonoid γ
        φ : (i : ι) → AddMonoidHom (β i) γ
        f : DFinsupp fun i => β i
        ⊢ ∀ (a b : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f.toFun i) …
      -/
      rintro ⟨sx, hx⟩ ⟨sy, hy⟩
      /-
        case mk.mk
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → AddZeroClass (β i)
        inst✝ : AddCommMonoid γ
        φ : (i : ι) → AddMonoidHom (β i) γ
        f : DFinsupp fun i => β i
        sx : Multiset ι
        hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
        sy : Multiset ι
        hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
        ⊢ Eq ((fun s => (↑s).toFinset.sum fun i => (φ i) (f i)) ⟨sx, hx⟩) ((fun s => ( …
      -/
      dsimp only [Subtype.coe_mk, toFun_eq_coe] at *
      /-
        case mk.mk
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → AddZeroClass (β i)
        inst✝ : AddCommMonoid γ
        φ : (i : ι) → AddMonoidHom (β i) γ
        f : DFinsupp fun i => β i
        sx : Multiset ι
        hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
        sy : Multiset ι
        hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
        ⊢ Eq (sx.toFinset.sum fun i => (φ i) (f i)) (sy.toFinset.sum fun i => (φ i) (f …
      -/
      have H1 : sx.toFinset ∩ sy.toFinset ⊆ sx.toFinset := Finset.inter_subset_left
      /-
        case mk.mk
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → AddZeroClass (β i)
        inst✝ : AddCommMonoid γ
        φ : (i : ι) → AddMonoidHom (β i) γ
        f : DFinsupp fun i => β i
        sx : Multiset ι
        hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
        sy : Multiset ι
        hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
        H1 : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sx.toFinset
        ⊢ Eq (sx.toFinset.sum fun i => (φ i) (f i)) (sy.toFinset.sum fun i => (φ i) (f …
      -/
      have H2 : sx.toFinset ∩ sy.toFinset ⊆ sy.toFinset := Finset.inter_subset_right
      refine
        (Finset.sum_subset H1 ?_).symm.trans
          ((Finset.sum_congr rfl ?_).trans (Finset.sum_subset H2 ?_))
        /-
          case mk.mk.refine_1
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : DFinsupp fun i => β i
          sx : Multiset ι
          hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
          sy : Multiset ι
          hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
          H1 : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sx.toFinset
          H2 : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sy.toFinset
          ⊢ ∀ (x : ι), Membership.mem sx.toFinset x → Not (Membership.mem (Inter.inter s …
        -/
      · intro i H1 H2
        /-
          case mk.mk.refine_1
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : DFinsupp fun i => β i
          sx : Multiset ι
          hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
          sy : Multiset ι
          hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
          H1✝ : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sx.toFinset
          H2✝ : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sy.toFinset
          i : ι
          H1 : Membership.mem sx.toFinset i
          H2 : Not (Membership.mem (Inter.inter sx.toFinset sy.toFinset) i)
          ⊢ Eq ((φ i) (f i)) 0
        -/
        rw [Finset.mem_inter] at H2
        /-
          case mk.mk.refine_1
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : DFinsupp fun i => β i
          sx : Multiset ι
          hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
          sy : Multiset ι
          hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
          H1✝ : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sx.toFinset
          H2✝ : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sy.toFinset
          i : ι
          H1 : Membership.mem sx.toFinset i
          H2 : Not (And (Membership.mem sx.toFinset i) (Membership.mem sy.toFinset i))
          ⊢ Eq ((φ i) (f i)) 0
        -/
        simp only [Multiset.mem_toFinset] at H1 H2
        /-
          case mk.mk.refine_1
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : DFinsupp fun i => β i
          sx : Multiset ι
          hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
          sy : Multiset ι
          hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
          H1✝ : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sx.toFinset
          H2✝ : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sy.toFinset
          i : ι
          H1 : Membership.mem sx i
          H2 : Not (And (Membership.mem sx i) (Membership.mem sy i))
          ⊢ Eq ((φ i) (f i)) 0
        -/
        convert AddMonoidHom.map_zero (φ i)
        /-
          case h.e'_2.h.e'_6
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : DFinsupp fun i => β i
          sx : Multiset ι
          hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
          sy : Multiset ι
          hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
          H1✝ : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sx.toFinset
          H2✝ : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sy.toFinset
          i : ι
          H1 : Membership.mem sx i
          H2 : Not (And (Membership.mem sx i) (Membership.mem sy i))
          ⊢ Eq (f i) 0
        -/
        exact (hy i).resolve_left (mt (And.intro H1) H2)
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.refine_2
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : DFinsupp fun i => β i
          sx : Multiset ι
          hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
          sy : Multiset ι
          hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
          H1 : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sx.toFinset
          H2 : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sy.toFinset
          ⊢ ∀ (x : ι), Membership.mem (Inter.inter sx.toFinset sy.toFinset) x → Eq ((φ x …
        -/
      · intro i _
        /-
          case mk.mk.refine_2
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : DFinsupp fun i => β i
          sx : Multiset ι
          hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
          sy : Multiset ι
          hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
          H1 : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sx.toFinset
          H2 : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sy.toFinset
          i : ι
          a✝ : Membership.mem (Inter.inter sx.toFinset sy.toFinset) i
          ⊢ Eq ((φ i) (f i)) ((φ i) (f i))
        -/
        rfl
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.refine_3
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : DFinsupp fun i => β i
          sx : Multiset ι
          hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
          sy : Multiset ι
          hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
          H1 : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sx.toFinset
          H2 : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sy.toFinset
          ⊢ ∀ (x : ι), Membership.mem sy.toFinset x → Not (Membership.mem (Inter.inter s …
        -/
      · intro i H1 H2
        /-
          case mk.mk.refine_3
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : DFinsupp fun i => β i
          sx : Multiset ι
          hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
          sy : Multiset ι
          hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
          H1✝ : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sx.toFinset
          H2✝ : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sy.toFinset
          i : ι
          H1 : Membership.mem sy.toFinset i
          H2 : Not (Membership.mem (Inter.inter sx.toFinset sy.toFinset) i)
          ⊢ Eq ((φ i) (f i)) 0
        -/
        rw [Finset.mem_inter] at H2
        /-
          case mk.mk.refine_3
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : DFinsupp fun i => β i
          sx : Multiset ι
          hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
          sy : Multiset ι
          hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
          H1✝ : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sx.toFinset
          H2✝ : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sy.toFinset
          i : ι
          H1 : Membership.mem sy.toFinset i
          H2 : Not (And (Membership.mem sx.toFinset i) (Membership.mem sy.toFinset i))
          ⊢ Eq ((φ i) (f i)) 0
        -/
        simp only [Multiset.mem_toFinset] at H1 H2
        /-
          case mk.mk.refine_3
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : DFinsupp fun i => β i
          sx : Multiset ι
          hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
          sy : Multiset ι
          hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
          H1✝ : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sx.toFinset
          H2✝ : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sy.toFinset
          i : ι
          H1 : Membership.mem sy i
          H2 : Not (And (Membership.mem sx i) (Membership.mem sy i))
          ⊢ Eq ((φ i) (f i)) 0
        -/
        convert AddMonoidHom.map_zero (φ i)
        /-
          case h.e'_2.h.e'_6
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : DFinsupp fun i => β i
          sx : Multiset ι
          hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
          sy : Multiset ι
          hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
          H1✝ : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sx.toFinset
          H2✝ : HasSubset.Subset (Inter.inter sx.toFinset sy.toFinset) sy.toFinset
          i : ι
          H1 : Membership.mem sy i
          H2 : Not (And (Membership.mem sx i) (Membership.mem sy i))
          ⊢ Eq (f i) 0
        -/
        exact (hx i).resolve_left (mt (fun H3 => And.intro H3 H1) H2)
        /-
          🎉 no goals
        -/
  map_add' := by
    /-
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → AddZeroClass (β i)
      inst✝ : AddCommMonoid γ
      φ : (i : ι) → AddMonoidHom (β i) γ
      ⊢ ∀ (x y : DFinsupp fun i => β i), Eq ({ toFun := fun f => (fun c => Trunc.lif …
    -/
    rintro ⟨f, sf, hf⟩ ⟨g, sg, hg⟩
    /-
      case mk'.mk.mk.mk'.mk.mk
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → AddZeroClass (β i)
      inst✝ : AddCommMonoid γ
      φ : (i : ι) → AddMonoidHom (β i) γ
      f : (i : ι) → β i
      support'✝¹ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f …
      sf : Multiset ι
      hf : ∀ (i : ι), Or (Membership.mem sf i) (Eq (f i) 0)
      g : (i : ι) → β i
      support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (g  …
      sg : Multiset ι
      hg : ∀ (i : ι), Or (Membership.mem sg i) (Eq (g i) 0)
      ⊢ Eq ({ toFun := fun f => (fun c => Trunc.lift (fun s => (↑s).toFinset.sum fun …
    -/
    change (∑ i ∈ _, _) = (∑ i ∈ _, _) + ∑ i ∈ _, _
    /-
      case mk'.mk.mk.mk'.mk.mk
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → AddZeroClass (β i)
      inst✝ : AddCommMonoid γ
      φ : (i : ι) → AddMonoidHom (β i) γ
      f : (i : ι) → β i
      support'✝¹ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f …
      sf : Multiset ι
      hf : ∀ (i : ι), Or (Membership.mem sf i) (Eq (f i) 0)
      g : (i : ι) → β i
      support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (g  …
      sg : Multiset ι
      hg : ∀ (i : ι), Or (Membership.mem sg i) (Eq (g i) 0)
      ⊢ Eq ((↑((fun ys => ⟨HAdd.hAdd ↑⟨sf, hf⟩ ↑ys, ⋯⟩) ⟨sg, hg⟩)).toFinset.sum fun  …
    -/
    simp only [coe_add, coe_mk', Subtype.coe_mk, Pi.add_apply, map_add, Finset.sum_add_distrib]
    /-
      case mk'.mk.mk.mk'.mk.mk
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → AddZeroClass (β i)
      inst✝ : AddCommMonoid γ
      φ : (i : ι) → AddMonoidHom (β i) γ
      f : (i : ι) → β i
      support'✝¹ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f …
      sf : Multiset ι
      hf : ∀ (i : ι), Or (Membership.mem sf i) (Eq (f i) 0)
      g : (i : ι) → β i
      support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (g  …
      sg : Multiset ι
      hg : ∀ (i : ι), Or (Membership.mem sg i) (Eq (g i) 0)
      ⊢ Eq (HAdd.hAdd ((HAdd.hAdd sf sg).toFinset.sum fun x => (φ x) (f x)) ((HAdd.h …
    -/
    congr 1
      /-
        case mk'.mk.mk.mk'.mk.mk.e_a
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → AddZeroClass (β i)
        inst✝ : AddCommMonoid γ
        φ : (i : ι) → AddMonoidHom (β i) γ
        f : (i : ι) → β i
        support'✝¹ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f …
        sf : Multiset ι
        hf : ∀ (i : ι), Or (Membership.mem sf i) (Eq (f i) 0)
        g : (i : ι) → β i
        support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (g  …
        sg : Multiset ι
        hg : ∀ (i : ι), Or (Membership.mem sg i) (Eq (g i) 0)
        ⊢ Eq ((HAdd.hAdd sf sg).toFinset.sum fun x => (φ x) (f x)) (sf.toFinset.sum fu …
      -/
    · refine (Finset.sum_subset ?_ ?_).symm
        /-
          case mk'.mk.mk.mk'.mk.mk.e_a.refine_1
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : (i : ι) → β i
          support'✝¹ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f …
          sf : Multiset ι
          hf : ∀ (i : ι), Or (Membership.mem sf i) (Eq (f i) 0)
          g : (i : ι) → β i
          support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (g  …
          sg : Multiset ι
          hg : ∀ (i : ι), Or (Membership.mem sg i) (Eq (g i) 0)
          ⊢ HasSubset.Subset sf.toFinset (HAdd.hAdd sf sg).toFinset
        -/
      · intro i
        /-
          case mk'.mk.mk.mk'.mk.mk.e_a.refine_1
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : (i : ι) → β i
          support'✝¹ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f …
          sf : Multiset ι
          hf : ∀ (i : ι), Or (Membership.mem sf i) (Eq (f i) 0)
          g : (i : ι) → β i
          support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (g  …
          sg : Multiset ι
          hg : ∀ (i : ι), Or (Membership.mem sg i) (Eq (g i) 0)
          i : ι
          ⊢ Membership.mem sf.toFinset i → Membership.mem (HAdd.hAdd sf sg).toFinset i
        -/
        simp only [Multiset.mem_toFinset, Multiset.mem_add]
        /-
          case mk'.mk.mk.mk'.mk.mk.e_a.refine_1
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : (i : ι) → β i
          support'✝¹ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f …
          sf : Multiset ι
          hf : ∀ (i : ι), Or (Membership.mem sf i) (Eq (f i) 0)
          g : (i : ι) → β i
          support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (g  …
          sg : Multiset ι
          hg : ∀ (i : ι), Or (Membership.mem sg i) (Eq (g i) 0)
          i : ι
          ⊢ Membership.mem sf i → Or (Membership.mem sf i) (Membership.mem sg i)
        -/
        exact Or.inl
        /-
          🎉 no goals
        -/
        /-
          case mk'.mk.mk.mk'.mk.mk.e_a.refine_2
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : (i : ι) → β i
          support'✝¹ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f …
          sf : Multiset ι
          hf : ∀ (i : ι), Or (Membership.mem sf i) (Eq (f i) 0)
          g : (i : ι) → β i
          support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (g  …
          sg : Multiset ι
          hg : ∀ (i : ι), Or (Membership.mem sg i) (Eq (g i) 0)
          ⊢ ∀ (x : ι), Membership.mem (HAdd.hAdd sf sg).toFinset x → Not (Membership.mem …
        -/
      · intro i _ H2
    /-
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → AddZeroClass (β i)
      inst✝ : AddCommMonoid γ
      φ : (i : ι) → AddMonoidHom (β i) γ
      ⊢ Eq ((fun f => (fun c => Trunc.lift (fun s => (↑s).toFinset.sum fun i => (φ i …
    -/
        /-
          case mk'.mk.mk.mk'.mk.mk.e_a.refine_2
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : (i : ι) → β i
          support'✝¹ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f …
          sf : Multiset ι
          hf : ∀ (i : ι), Or (Membership.mem sf i) (Eq (f i) 0)
          g : (i : ι) → β i
          support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (g  …
          sg : Multiset ι
          hg : ∀ (i : ι), Or (Membership.mem sg i) (Eq (g i) 0)
          i : ι
          a✝ : Membership.mem (HAdd.hAdd sf sg).toFinset i
          H2 : Not (Membership.mem sf.toFinset i)
          ⊢ Eq ((φ i) (f i)) 0
        -/
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/
        simp only [Multiset.mem_toFinset, Multiset.mem_add] at H2
        /-
          case mk'.mk.mk.mk'.mk.mk.e_a.refine_2
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : (i : ι) → β i
          support'✝¹ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f …
          sf : Multiset ι
          hf : ∀ (i : ι), Or (Membership.mem sf i) (Eq (f i) 0)
          g : (i : ι) → β i
          support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (g  …
          sg : Multiset ι
          hg : ∀ (i : ι), Or (Membership.mem sg i) (Eq (g i) 0)
          i : ι
          a✝ : Membership.mem (HAdd.hAdd sf sg).toFinset i
          H2 : Not (Membership.mem sf i)
          ⊢ Eq ((φ i) (f i)) 0
        -/
        rw [(hf i).resolve_left H2, AddMonoidHom.map_zero]
        /-
          🎉 no goals
        -/
      /-
        case mk'.mk.mk.mk'.mk.mk.e_a
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → AddZeroClass (β i)
        inst✝ : AddCommMonoid γ
        φ : (i : ι) → AddMonoidHom (β i) γ
        f : (i : ι) → β i
        support'✝¹ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f …
        sf : Multiset ι
        hf : ∀ (i : ι), Or (Membership.mem sf i) (Eq (f i) 0)
        g : (i : ι) → β i
        support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (g  …
        sg : Multiset ι
        hg : ∀ (i : ι), Or (Membership.mem sg i) (Eq (g i) 0)
        ⊢ Eq ((HAdd.hAdd sf sg).toFinset.sum fun x => (φ x) (g x)) (sg.toFinset.sum fu …
      -/
    · refine (Finset.sum_subset ?_ ?_).symm
        /-
          case mk'.mk.mk.mk'.mk.mk.e_a.refine_1
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : (i : ι) → β i
          support'✝¹ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f …
          sf : Multiset ι
          hf : ∀ (i : ι), Or (Membership.mem sf i) (Eq (f i) 0)
          g : (i : ι) → β i
          support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (g  …
          sg : Multiset ι
          hg : ∀ (i : ι), Or (Membership.mem sg i) (Eq (g i) 0)
          ⊢ HasSubset.Subset sg.toFinset (HAdd.hAdd sf sg).toFinset
        -/
      · intro i
        /-
          case mk'.mk.mk.mk'.mk.mk.e_a.refine_1
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : (i : ι) → β i
          support'✝¹ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f …
          sf : Multiset ι
          hf : ∀ (i : ι), Or (Membership.mem sf i) (Eq (f i) 0)
          g : (i : ι) → β i
          support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (g  …
          sg : Multiset ι
          hg : ∀ (i : ι), Or (Membership.mem sg i) (Eq (g i) 0)
          i : ι
          ⊢ Membership.mem sg.toFinset i → Membership.mem (HAdd.hAdd sf sg).toFinset i
        -/
        simp only [Multiset.mem_toFinset, Multiset.mem_add]
        /-
          case mk'.mk.mk.mk'.mk.mk.e_a.refine_1
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : (i : ι) → β i
          support'✝¹ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f …
          sf : Multiset ι
          hf : ∀ (i : ι), Or (Membership.mem sf i) (Eq (f i) 0)
          g : (i : ι) → β i
          support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (g  …
          sg : Multiset ι
          hg : ∀ (i : ι), Or (Membership.mem sg i) (Eq (g i) 0)
          i : ι
          ⊢ Membership.mem sg i → Or (Membership.mem sf i) (Membership.mem sg i)
        -/
        exact Or.inr
        /-
          🎉 no goals
        -/
        /-
          case mk'.mk.mk.mk'.mk.mk.e_a.refine_2
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : (i : ι) → β i
          support'✝¹ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f …
          sf : Multiset ι
          hf : ∀ (i : ι), Or (Membership.mem sf i) (Eq (f i) 0)
          g : (i : ι) → β i
          support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (g  …
          sg : Multiset ι
          hg : ∀ (i : ι), Or (Membership.mem sg i) (Eq (g i) 0)
          ⊢ ∀ (x : ι), Membership.mem (HAdd.hAdd sf sg).toFinset x → Not (Membership.mem …
        -/
      · intro i _ H2
        /-
          case mk'.mk.mk.mk'.mk.mk.e_a.refine_2
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : (i : ι) → β i
          support'✝¹ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f …
          sf : Multiset ι
          hf : ∀ (i : ι), Or (Membership.mem sf i) (Eq (f i) 0)
          g : (i : ι) → β i
          support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (g  …
          sg : Multiset ι
          hg : ∀ (i : ι), Or (Membership.mem sg i) (Eq (g i) 0)
          i : ι
          a✝ : Membership.mem (HAdd.hAdd sf sg).toFinset i
          H2 : Not (Membership.mem sg.toFinset i)
          ⊢ Eq ((φ i) (g i)) 0
        -/
        simp only [Multiset.mem_toFinset, Multiset.mem_add] at H2
        /-
          case mk'.mk.mk.mk'.mk.mk.e_a.refine_2
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → AddZeroClass (β i)
          inst✝ : AddCommMonoid γ
          φ : (i : ι) → AddMonoidHom (β i) γ
          f : (i : ι) → β i
          support'✝¹ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f …
          sf : Multiset ι
          hf : ∀ (i : ι), Or (Membership.mem sf i) (Eq (f i) 0)
          g : (i : ι) → β i
          support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (g  …
          sg : Multiset ι
          hg : ∀ (i : ι), Or (Membership.mem sg i) (Eq (g i) 0)
          i : ι
          a✝ : Membership.mem (HAdd.hAdd sf sg).toFinset i
          H2 : Not (Membership.mem sg i)
          ⊢ Eq ((φ i) (g i)) 0
        -/
        rw [(hg i).resolve_left H2, AddMonoidHom.map_zero]
        /-
          🎉 no goals
        -/
  map_zero' := by
    simp only [toFun_eq_coe, coe_zero, Pi.zero_apply, map_zero, Finset.sum_const_zero]; rfl


@[simp]
theorem sumAddHom_single [∀ i, AddZeroClass (β i)] [AddCommMonoid γ] (φ : ∀ i, β i →+ γ) (i)
    (x : β i) : sumAddHom φ (single i x) = φ i x := by
  /-
    ι : Type u
    γ : Type w
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → AddZeroClass (β i)
    inst✝ : AddCommMonoid γ
    φ : (i : ι) → AddMonoidHom (β i) γ
    i : ι
    x : β i
    ⊢ Eq ((DFinsupp.sumAddHom φ) (DFinsupp.single i x)) ((φ i) x)
  -/
  dsimp [sumAddHom, single, Trunc.lift_mk]
  /-
    ι : Type u
    γ : Type w
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → AddZeroClass (β i)
    inst✝ : AddCommMonoid γ
    φ : (i : ι) → AddMonoidHom (β i) γ
    i : ι
    x : β i
    ⊢ Eq ((Singleton.singleton i).toFinset.sum fun i_1 => (φ i_1) (Pi.single i x i …
  -/
  rw [Multiset.toFinset_singleton, Finset.sum_singleton, Pi.single_eq_same]
  /-
    🎉 no goals
  -/


@[simp]
theorem sumAddHom_comp_single [∀ i, AddZeroClass (β i)] [AddCommMonoid γ] (f : ∀ i, β i →+ γ)
    (i : ι) : (sumAddHom f).comp (singleAddHom β i) = f i :=
  AddMonoidHom.ext fun x => sumAddHom_single f i x


/-- While we didn't need decidable instances to define it, we do to reduce it to a sum -/
theorem sumAddHom_apply [∀ i, AddZeroClass (β i)] [∀ (i) (x : β i), Decidable (x ≠ 0)]
    [AddCommMonoid γ] (φ : ∀ i, β i →+ γ) (f : Π₀ i, β i) : sumAddHom φ f = f.sum fun x => φ x := by
  /-
    ι : Type u
    γ : Type w
    β : ι → Type v
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → AddZeroClass (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    inst✝ : AddCommMonoid γ
    φ : (i : ι) → AddMonoidHom (β i) γ
    f : DFinsupp fun i => β i
    ⊢ Eq ((DFinsupp.sumAddHom φ) f) (f.sum fun x => ⇑(φ x))
  -/
  rcases f with ⟨f, s, hf⟩
  /-
    case mk'.mk.mk
    ι : Type u
    γ : Type w
    β : ι → Type v
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → AddZeroClass (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    inst✝ : AddCommMonoid γ
    φ : (i : ι) → AddMonoidHom (β i) γ
    f : (i : ι) → β i
    support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f  …
    s : Multiset ι
    hf : ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0)
    ⊢ Eq ((DFinsupp.sumAddHom φ) { toFun := f, support' := Quot.mk ⇑trueSetoid ⟨s, …
  -/
  change (∑ i ∈ _, _) = ∑ i ∈ _ with _, _
  /-
    case mk'.mk.mk
    ι : Type u
    γ : Type w
    β : ι → Type v
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → AddZeroClass (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    inst✝ : AddCommMonoid γ
    φ : (i : ι) → AddMonoidHom (β i) γ
    f : (i : ι) → β i
    support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f  …
    s : Multiset ι
    hf : ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0)
    ⊢ Eq ((↑⟨s, hf⟩).toFinset.sum fun i => (φ i) ({ toFun := f, support' := Quot.m …
  -/
  rw [Finset.sum_filter, Finset.sum_congr rfl]
  /-
    case mk'.mk.mk
    ι : Type u
    γ : Type w
    β : ι → Type v
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → AddZeroClass (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    inst✝ : AddCommMonoid γ
    φ : (i : ι) → AddMonoidHom (β i) γ
    f : (i : ι) → β i
    support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f  …
    s : Multiset ι
    hf : ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0)
    ⊢ ∀ (x : ι), Membership.mem (↑⟨s, hf⟩).toFinset x → Eq ((φ x) ({ toFun := f, s …
  -/
  intro i _
  /-
    case mk'.mk.mk
    ι : Type u
    γ : Type w
    β : ι → Type v
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → AddZeroClass (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    inst✝ : AddCommMonoid γ
    φ : (i : ι) → AddMonoidHom (β i) γ
    f : (i : ι) → β i
    support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f  …
    s : Multiset ι
    hf : ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0)
    i : ι
    a✝ : Membership.mem (↑⟨s, hf⟩).toFinset i
    ⊢ Eq ((φ i) ({ toFun := f, support' := Quot.mk ⇑trueSetoid ⟨s, hf⟩ } i)) (ite  …
  -/
  dsimp only [coe_mk', Subtype.coe_mk] at *
  /-
    case mk'.mk.mk
    ι : Type u
    γ : Type w
    β : ι → Type v
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → AddZeroClass (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    inst✝ : AddCommMonoid γ
    φ : (i : ι) → AddMonoidHom (β i) γ
    f : (i : ι) → β i
    support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f  …
    s : Multiset ι
    hf : ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0)
    i : ι
    a✝ : Membership.mem s.toFinset i
    ⊢ Eq ((φ i) (f i)) (ite (Ne (f i) 0) ((φ i) (f i)) 0)
  -/
  split_ifs with h
    /-
      case pos
      ι : Type u
      γ : Type w
      β : ι → Type v
      inst✝³ : DecidableEq ι
      inst✝² : (i : ι) → AddZeroClass (β i)
      inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      inst✝ : AddCommMonoid γ
      φ : (i : ι) → AddMonoidHom (β i) γ
      f : (i : ι) → β i
      support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f  …
      s : Multiset ι
      hf : ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0)
      i : ι
      a✝ : Membership.mem s.toFinset i
      h : Ne (f i) 0
      ⊢ Eq ((φ i) (f i)) ((φ i) (f i))
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u
      γ : Type w
      β : ι → Type v
      inst✝³ : DecidableEq ι
      inst✝² : (i : ι) → AddZeroClass (β i)
      inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      inst✝ : AddCommMonoid γ
      φ : (i : ι) → AddMonoidHom (β i) γ
      f : (i : ι) → β i
      support'✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f  …
      s : Multiset ι
      hf : ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0)
      i : ι
      a✝ : Membership.mem s.toFinset i
      h : Not (Ne (f i) 0)
      ⊢ Eq ((φ i) (f i)) 0
    -/
  · rw [not_not.mp h, AddMonoidHom.map_zero]
    /-
      🎉 no goals
    -/


theorem sumAddHom_comm {ι₁ ι₂ : Sort _} {β₁ : ι₁ → Type*} {β₂ : ι₂ → Type*} {γ : Type*}
    [DecidableEq ι₁] [DecidableEq ι₂] [∀ i, AddZeroClass (β₁ i)] [∀ i, AddZeroClass (β₂ i)]
    [AddCommMonoid γ] (f₁ : Π₀ i, β₁ i) (f₂ : Π₀ i, β₂ i) (h : ∀ i j, β₁ i →+ β₂ j →+ γ) :
    sumAddHom (fun i₂ => sumAddHom (fun i₁ => h i₁ i₂) f₁) f₂ =
      sumAddHom (fun i₁ => sumAddHom (fun i₂ => (h i₁ i₂).flip) f₂) f₁ := by
  /-
    ι₁ : Type u_4
    ι₂ : Type u_5
    β₁ : ι₁ → Type u_1
    β₂ : ι₂ → Type u_2
    γ : Type u_3
    inst✝⁴ : DecidableEq ι₁
    inst✝³ : DecidableEq ι₂
    inst✝² : (i : ι₁) → AddZeroClass (β₁ i)
    inst✝¹ : (i : ι₂) → AddZeroClass (β₂ i)
    inst✝ : AddCommMonoid γ
    f₁ : DFinsupp fun i => β₁ i
    f₂ : DFinsupp fun i => β₂ i
    h : (i : ι₁) → (j : ι₂) → AddMonoidHom (β₁ i) (AddMonoidHom (β₂ j) γ)
    ⊢ Eq ((DFinsupp.sumAddHom fun i₂ => (DFinsupp.sumAddHom fun i₁ => h i₁ i₂) f₁) …
  -/
  obtain ⟨⟨f₁, s₁, h₁⟩, ⟨f₂, s₂, h₂⟩⟩ := f₁, f₂
  simp only [sumAddHom, AddMonoidHom.finset_sum_apply, Quotient.liftOn_mk, AddMonoidHom.coe_mk,
    AddMonoidHom.flip_apply, Trunc.lift, toFun_eq_coe, ZeroHom.coe_mk, coe_mk']
  /-
    case mk'.mk.mk.mk'.mk.mk
    ι₁ : Type u_4
    ι₂ : Type u_5
    β₁ : ι₁ → Type u_1
    β₂ : ι₂ → Type u_2
    γ : Type u_3
    inst✝⁴ : DecidableEq ι₁
    inst✝³ : DecidableEq ι₂
    inst✝² : (i : ι₁) → AddZeroClass (β₁ i)
    inst✝¹ : (i : ι₂) → AddZeroClass (β₂ i)
    inst✝ : AddCommMonoid γ
    h : (i : ι₁) → (j : ι₂) → AddMonoidHom (β₁ i) (AddMonoidHom (β₂ j) γ)
    f₁ : (i : ι₁) → β₁ i
    support'✝¹ : Trunc (Subtype fun s => ∀ (i : ι₁), Or (Membership.mem s i) (Eq ( …
    s₁ : Multiset ι₁
    h₁ : ∀ (i : ι₁), Or (Membership.mem s₁ i) (Eq (f₁ i) 0)
    f₂ : (i : ι₂) → β₂ i
    support'✝ : Trunc (Subtype fun s => ∀ (i : ι₂), Or (Membership.mem s i) (Eq (f …
    s₂ : Multiset ι₂
    h₂ : ∀ (i : ι₂), Or (Membership.mem s₂ i) (Eq (f₂ i) 0)
    ⊢ Eq (s₂.toFinset.sum fun x => s₁.toFinset.sum fun x_1 => ((h x_1 x) (f₁ x_1)) …
  -/
  exact Finset.sum_comm
  /-
    🎉 no goals
  -/


/-- The `DFinsupp` version of `Finsupp.liftAddHom`,-/
@[simps apply symm_apply]
def liftAddHom [∀ i, AddZeroClass (β i)] [AddCommMonoid γ] :
    (∀ i, β i →+ γ) ≃+ ((Π₀ i, β i) →+ γ) where
  toFun := sumAddHom
  invFun F i := F.comp (singleAddHom β i)
                   /-
                     ι : Type u
                     γ : Type w
                     β : ι → Type v
                     β₁ : ι → Type v₁
                     β₂ : ι → Type v₂
                     inst✝² : DecidableEq ι
                     inst✝¹ : (i : ι) → AddZeroClass (β i)
                     inst✝ : AddCommMonoid γ
                     x : (i : ι) → AddMonoidHom (β i) γ
                     ⊢ Eq ((fun F i => F.comp (DFinsupp.singleAddHom β i)) (DFinsupp.sumAddHom x)) x
                   -/
  left_inv x := by ext; simp
                        /-
                          🎉 no goals
                        -/
                    /-
                      ι : Type u
                      γ : Type w
                      β : ι → Type v
                      β₁ : ι → Type v₁
                      β₂ : ι → Type v₂
                      inst✝² : DecidableEq ι
                      inst✝¹ : (i : ι) → AddZeroClass (β i)
                      inst✝ : AddCommMonoid γ
                      ψ : AddMonoidHom (DFinsupp fun i => β i) γ
                      ⊢ Eq (DFinsupp.sumAddHom ((fun F i => F.comp (DFinsupp.singleAddHom β i)) ψ)) ψ
                    -/
  right_inv ψ := by ext; simp
                         /-
                           🎉 no goals
                         -/
                     /-
                       ι : Type u
                       γ : Type w
                       β : ι → Type v
                       β₁ : ι → Type v₁
                       β₂ : ι → Type v₂
                       inst✝² : DecidableEq ι
                       inst✝¹ : (i : ι) → AddZeroClass (β i)
                       inst✝ : AddCommMonoid γ
                       F G : (i : ι) → AddMonoidHom (β i) γ
                       ⊢ Eq ({ toFun := DFinsupp.sumAddHom, invFun := fun F i => F.comp (DFinsupp.sin …
                     -/
  map_add' F G := by ext; simp
                          /-
                            🎉 no goals
                          -/

-- Porting note: The elaborator is struggling with `liftAddHom`. Passing it `β` explicitly helps.
-- This applies to roughly the remainder of the file.


/-- The `DFinsupp` version of `Finsupp.liftAddHom_singleAddHom`,-/
@[simp, nolint simpNF] -- Porting note: linter claims that simp can prove this, but it can not
theorem liftAddHom_singleAddHom [∀ i, AddCommMonoid (β i)] :
    liftAddHom (β := β) (singleAddHom β) = AddMonoidHom.id (Π₀ i, β i) :=
  (liftAddHom (β := β)).toEquiv.apply_eq_iff_eq_symm_apply.2 rfl


/-- The `DFinsupp` version of `Finsupp.liftAddHom_apply_single`,-/
theorem liftAddHom_apply_single [∀ i, AddZeroClass (β i)] [AddCommMonoid γ] (f : ∀ i, β i →+ γ)
                                                                         /-
                                                                           ι : Type u
                                                                           γ : Type w
                                                                           β : ι → Type v
                                                                           inst✝² : DecidableEq ι
                                                                           inst✝¹ : (i : ι) → AddZeroClass (β i)
                                                                           inst✝ : AddCommMonoid γ
                                                                           f : (i : ι) → AddMonoidHom (β i) γ
                                                                           i : ι
                                                                           x : β i
                                                                           ⊢ Eq ((DFinsupp.liftAddHom f) (DFinsupp.single i x)) ((f i) x)
                                                                         -/
    (i : ι) (x : β i) : liftAddHom (β := β) f (single i x) = f i x := by simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- The `DFinsupp` version of `Finsupp.liftAddHom_comp_single`,-/
theorem liftAddHom_comp_single [∀ i, AddZeroClass (β i)] [AddCommMonoid γ] (f : ∀ i, β i →+ γ)
                                                                          /-
                                                                            ι : Type u
                                                                            γ : Type w
                                                                            β : ι → Type v
                                                                            inst✝² : DecidableEq ι
                                                                            inst✝¹ : (i : ι) → AddZeroClass (β i)
                                                                            inst✝ : AddCommMonoid γ
                                                                            f : (i : ι) → AddMonoidHom (β i) γ
                                                                            i : ι
                                                                            ⊢ Eq ((DFinsupp.liftAddHom f).comp (DFinsupp.singleAddHom β i)) (f i)
                                                                          -/
    (i : ι) : (liftAddHom (β := β) f).comp (singleAddHom β i) = f i := by simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


/-- The `DFinsupp` version of `Finsupp.comp_liftAddHom`,-/
theorem comp_liftAddHom {δ : Type*} [∀ i, AddZeroClass (β i)] [AddCommMonoid γ] [AddCommMonoid δ]
    (g : γ →+ δ) (f : ∀ i, β i →+ γ) :
    g.comp (liftAddHom (β := β) f) = liftAddHom (β := β) fun a => g.comp (f a) :=
  (liftAddHom (β := β)).symm_apply_eq.1 <|
    funext fun a => by
      /-
        ι : Type u
        γ : Type w
        β : ι → Type v
        inst✝³ : DecidableEq ι
        δ : Type u_1
        inst✝² : (i : ι) → AddZeroClass (β i)
        inst✝¹ : AddCommMonoid γ
        inst✝ : AddCommMonoid δ
        g : AddMonoidHom γ δ
        f : (i : ι) → AddMonoidHom (β i) γ
        a : ι
        ⊢ Eq (DFinsupp.liftAddHom.symm (g.comp (DFinsupp.liftAddHom f)) a) (g.comp (f  …
      -/
      rw [liftAddHom_symm_apply, AddMonoidHom.comp_assoc, liftAddHom_comp_single]
      /-
        🎉 no goals
      -/


@[simp]
theorem sumAddHom_zero [∀ i, AddZeroClass (β i)] [AddCommMonoid γ] :
    (sumAddHom fun i => (0 : β i →+ γ)) = 0 :=
  map_zero (liftAddHom (β := β))


@[simp]
theorem sumAddHom_add [∀ i, AddZeroClass (β i)] [AddCommMonoid γ] (g : ∀ i, β i →+ γ)
    (h : ∀ i, β i →+ γ) : (sumAddHom fun i => g i + h i) = sumAddHom g + sumAddHom h :=
  map_add (liftAddHom (β := β)) _ _


@[simp]
theorem sumAddHom_singleAddHom [∀ i, AddCommMonoid (β i)] :
    sumAddHom (singleAddHom β) = AddMonoidHom.id _ :=
  liftAddHom_singleAddHom


theorem comp_sumAddHom {δ : Type*} [∀ i, AddZeroClass (β i)] [AddCommMonoid γ] [AddCommMonoid δ]
    (g : γ →+ δ) (f : ∀ i, β i →+ γ) : g.comp (sumAddHom f) = sumAddHom fun a => g.comp (f a) :=
  comp_liftAddHom _ _


theorem sum_sub_index [∀ i, AddGroup (β i)] [∀ (i) (x : β i), Decidable (x ≠ 0)] [AddCommGroup γ]
    {f g : Π₀ i, β i} {h : ∀ i, β i → γ} (h_sub : ∀ i b₁ b₂, h i (b₁ - b₂) = h i b₁ - h i b₂) :
    (f - g).sum h = f.sum h - g.sum h := by
  /-
    ι : Type u
    γ : Type w
    β : ι → Type v
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → AddGroup (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    inst✝ : AddCommGroup γ
    f g : DFinsupp fun i => β i
    h : (i : ι) → β i → γ
    h_sub : ∀ (i : ι) (b₁ b₂ : β i), Eq (h i (HSub.hSub b₁ b₂)) (HSub.hSub (h i b₁ …
    ⊢ Eq ((HSub.hSub f g).sum h) (HSub.hSub (f.sum h) (g.sum h))
  -/
  have := (liftAddHom (β := β) fun a => AddMonoidHom.ofMapSub (h a) (h_sub a)).map_sub f g
  /-
    ι : Type u
    γ : Type w
    β : ι → Type v
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → AddGroup (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    inst✝ : AddCommGroup γ
    f g : DFinsupp fun i => β i
    h : (i : ι) → β i → γ
    h_sub : ∀ (i : ι) (b₁ b₂ : β i), Eq (h i (HSub.hSub b₁ b₂)) (HSub.hSub (h i b₁ …
    this : Eq ((DFinsupp.liftAddHom fun a => AddMonoidHom.ofMapSub (h a) ⋯) (HSub. …
    ⊢ Eq ((HSub.hSub f g).sum h) (HSub.hSub (f.sum h) (g.sum h))
  -/
  rw [liftAddHom_apply, sumAddHom_apply, sumAddHom_apply, sumAddHom_apply] at this
  /-
    ι : Type u
    γ : Type w
    β : ι → Type v
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → AddGroup (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    inst✝ : AddCommGroup γ
    f g : DFinsupp fun i => β i
    h : (i : ι) → β i → γ
    h_sub : ∀ (i : ι) (b₁ b₂ : β i), Eq (h i (HSub.hSub b₁ b₂)) (HSub.hSub (h i b₁ …
    this : Eq ((HSub.hSub f g).sum fun x => ⇑(AddMonoidHom.ofMapSub (h x) ⋯)) (HSu …
    ⊢ Eq ((HSub.hSub f g).sum h) (HSub.hSub (f.sum h) (g.sum h))
  -/
  exact this
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_finset_sum_index {γ : Type w} {α : Type x} [∀ i, AddCommMonoid (β i)]
    [∀ (i) (x : β i), Decidable (x ≠ 0)] [CommMonoid γ] {s : Finset α} {g : α → Π₀ i, β i}
    {h : ∀ i, β i → γ} (h_zero : ∀ i, h i 0 = 1)
    (h_add : ∀ i b₁ b₂, h i (b₁ + b₂) = h i b₁ * h i b₂) :
    (∏ i ∈ s, (g i).prod h) = (∑ i ∈ s, g i).prod h := by
  classical
  exact Finset.induction_on s (by simp [prod_zero_index])
        (by simp +contextual [prod_add_index, h_zero, h_add])


@[to_additive]
theorem prod_sum_index {ι₁ : Type u₁} [DecidableEq ι₁] {β₁ : ι₁ → Type v₁} [∀ i₁, Zero (β₁ i₁)]
    [∀ (i) (x : β₁ i), Decidable (x ≠ 0)] [∀ i, AddCommMonoid (β i)]
    [∀ (i) (x : β i), Decidable (x ≠ 0)] [CommMonoid γ] {f : Π₀ i₁, β₁ i₁}
    {g : ∀ i₁, β₁ i₁ → Π₀ i, β i} {h : ∀ i, β i → γ} (h_zero : ∀ i, h i 0 = 1)
    (h_add : ∀ i b₁ b₂, h i (b₁ + b₂) = h i b₁ * h i b₂) :
    (f.sum g).prod h = f.prod fun i b => (g i b).prod h :=
  (prod_finset_sum_index h_zero h_add).symm


@[simp]
theorem sum_single [∀ i, AddCommMonoid (β i)] [∀ (i) (x : β i), Decidable (x ≠ 0)] {f : Π₀ i, β i} :
    f.sum single = f := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → AddCommMonoid (β i)
    inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    f : DFinsupp fun i => β i
    ⊢ Eq (f.sum DFinsupp.single) f
  -/
  have := DFunLike.congr_fun (liftAddHom_singleAddHom (β := β)) f
  /-
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → AddCommMonoid (β i)
    inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    f : DFinsupp fun i => β i
    this : Eq ((DFinsupp.liftAddHom (DFinsupp.singleAddHom β)) f) ((AddMonoidHom.i …
    ⊢ Eq (f.sum DFinsupp.single) f
  -/
  rw [liftAddHom_apply, sumAddHom_apply] at this
  /-
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → AddCommMonoid (β i)
    inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    f : DFinsupp fun i => β i
    this : Eq (f.sum fun x => ⇑(DFinsupp.singleAddHom β x)) ((AddMonoidHom.id (DFi …
    ⊢ Eq (f.sum DFinsupp.single) f
  -/
  exact this
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_subtypeDomain_index [∀ i, Zero (β i)] [∀ (i) (x : β i), Decidable (x ≠ 0)]
    [CommMonoid γ] {v : Π₀ i, β i} {p : ι → Prop} [DecidablePred p] {h : ∀ i, β i → γ}
    (hp : ∀ x ∈ v.support, p x) : (v.subtypeDomain p).prod (fun i b => h i b) = v.prod h := by
  /-
    ι : Type u
    γ : Type w
    β : ι → Type v
    inst✝⁴ : DecidableEq ι
    inst✝³ : (i : ι) → Zero (β i)
    inst✝² : (i : ι) → (x : β i) → Decidable (Ne x 0)
    inst✝¹ : CommMonoid γ
    v : DFinsupp fun i => β i
    p : ι → Prop
    inst✝ : DecidablePred p
    h : (i : ι) → β i → γ
    hp : ∀ (x : ι), Membership.mem v.support x → p x
    ⊢ Eq ((DFinsupp.subtypeDomain p v).prod fun i b => h (↑i) b) (v.prod h)
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
  refine Finset.prod_bij (fun p _ ↦ p) ?_ ?_ ?_ ?_ <;> aesop
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem subtypeDomain_sum {ι} {β : ι → Type v} [∀ i, AddCommMonoid (β i)] {s : Finset γ}
    {h : γ → Π₀ i, β i} {p : ι → Prop} [DecidablePred p] :
    (∑ c ∈ s, h c).subtypeDomain p = ∑ c ∈ s, (h c).subtypeDomain p :=
  map_sum (subtypeDomainAddMonoidHom β p) _ s


theorem subtypeDomain_finsupp_sum {ι} {β : ι → Type v} {δ : γ → Type x} [DecidableEq γ]
    [∀ c, Zero (δ c)]  [∀ (c) (x : δ c), Decidable (x ≠ 0)]
    [∀ i, AddCommMonoid (β i)] {p : ι → Prop} [DecidablePred p]
    {s : Π₀ c, δ c} {h : ∀ c, δ c → Π₀ i, β i} :
    (s.sum h).subtypeDomain p = s.sum fun c d => (h c d).subtypeDomain p :=
  subtypeDomain_sum


@[to_additive (attr := simp, norm_cast)]
theorem coe_dfinsupp_prod [Monoid R] [CommMonoid S] (f : Π₀ i, β i) (g : ∀ i, β i → R →* S) :
    ⇑(f.prod g) = f.prod fun a b => ⇑(g a b) :=
  coe_finset_prod _ _


@[to_additive]
theorem dfinsupp_prod_apply [Monoid R] [CommMonoid S] (f : Π₀ i, β i) (g : ∀ i, β i → R →* S)
    (r : R) : (f.prod g) r = f.prod fun a b => (g a b) r :=
  finset_prod_apply _ _ _


@[simp]
theorem map_dfinsupp_sumAddHom [AddCommMonoid R] [AddCommMonoid S] [∀ i, AddZeroClass (β i)]
    (h : R →+ S) (f : Π₀ i, β i) (g : ∀ i, β i →+ R) :
    h (sumAddHom g f) = sumAddHom (fun i => h.comp (g i)) f :=
  DFunLike.congr_fun (comp_liftAddHom h g) f


theorem dfinsupp_sumAddHom_apply [AddZeroClass R] [AddCommMonoid S] [∀ i, AddZeroClass (β i)]
    (f : Π₀ i, β i) (g : ∀ i, β i →+ R →+ S) (r : R) :
    (sumAddHom g f) r = sumAddHom (fun i => (eval r).comp (g i)) f :=
  map_dfinsupp_sumAddHom (eval r) f g


@[simp, norm_cast]
theorem coe_dfinsupp_sumAddHom [AddZeroClass R] [AddCommMonoid S] [∀ i, AddZeroClass (β i)]
    (f : Π₀ i, β i) (g : ∀ i, β i →+ R →+ S) :
    ⇑(sumAddHom g f) = sumAddHom (fun i => (coeFn R S).comp (g i)) f :=
  map_dfinsupp_sumAddHom (coeFn R S) f g


@[simp]
theorem map_dfinsupp_sumAddHom [NonAssocSemiring R] [NonAssocSemiring S] [∀ i, AddZeroClass (β i)]
    (h : R →+* S) (f : Π₀ i, β i) (g : ∀ i, β i →+ R) :
    h (sumAddHom g f) = sumAddHom (fun i => h.toAddMonoidHom.comp (g i)) f :=
  DFunLike.congr_fun (comp_liftAddHom h.toAddMonoidHom g) f


@[simp]
theorem map_dfinsupp_sumAddHom [AddCommMonoid R] [AddCommMonoid S] [∀ i, AddZeroClass (β i)]
    (h : R ≃+ S) (f : Π₀ i, β i) (g : ∀ i, β i →+ R) :
    h (sumAddHom g f) = sumAddHom (fun i => h.toAddMonoidHom.comp (g i)) f :=
  DFunLike.congr_fun (comp_liftAddHom h.toAddMonoidHom g) f


