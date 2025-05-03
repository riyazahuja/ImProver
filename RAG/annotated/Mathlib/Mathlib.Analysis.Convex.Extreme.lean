/-- A set `B` is an extreme subset of `A` if `B ⊆ A` and all points of `B` only belong to open
segments whose ends are in `B`. -/
def IsExtreme (A B : Set E) : Prop :=
  B ⊆ A ∧ ∀ ⦃x₁⦄, x₁ ∈ A → ∀ ⦃x₂⦄, x₂ ∈ A → ∀ ⦃x⦄, x ∈ B → x ∈ openSegment 𝕜 x₁ x₂ → x₁ ∈ B ∧ x₂ ∈ B


/-- A point `x` is an extreme point of a set `A` if `x` belongs to no open segment with ends in
`A`, except for the obvious `openSegment x x`. -/
def Set.extremePoints (A : Set E) : Set E :=
  { x ∈ A | ∀ ⦃x₁⦄, x₁ ∈ A → ∀ ⦃x₂⦄, x₂ ∈ A → x ∈ openSegment 𝕜 x₁ x₂ → x₁ = x ∧ x₂ = x }


@[refl]
protected theorem IsExtreme.refl (A : Set E) : IsExtreme 𝕜 A A :=
  ⟨Subset.rfl, fun _ hx₁A _ hx₂A _ _ _ ↦ ⟨hx₁A, hx₂A⟩⟩


protected theorem IsExtreme.rfl : IsExtreme 𝕜 A A :=
  IsExtreme.refl 𝕜 A


@[trans]
protected theorem IsExtreme.trans (hAB : IsExtreme 𝕜 A B) (hBC : IsExtreme 𝕜 B C) :
    IsExtreme 𝕜 A C := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    A B C : Set E
    hAB : IsExtreme 𝕜 A B
    hBC : IsExtreme 𝕜 B C
    ⊢ IsExtreme 𝕜 A C
  -/
  refine ⟨Subset.trans hBC.1 hAB.1, fun x₁ hx₁A x₂ hx₂A x hxC hx ↦ ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    A B C : Set E
    hAB : IsExtreme 𝕜 A B
    hBC : IsExtreme 𝕜 B C
    x₁ : E
    hx₁A : Membership.mem A x₁
    x₂ : E
    hx₂A : Membership.mem A x₂
    x : E
    hxC : Membership.mem C x
    hx : Membership.mem (openSegment 𝕜 x₁ x₂) x
    ⊢ And (Membership.mem C x₁) (Membership.mem C x₂)
  -/
  obtain ⟨hx₁B, hx₂B⟩ := hAB.2 hx₁A hx₂A (hBC.1 hxC) hx
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    A B C : Set E
    hAB : IsExtreme 𝕜 A B
    hBC : IsExtreme 𝕜 B C
    x₁ : E
    hx₁A : Membership.mem A x₁
    x₂ : E
    hx₂A : Membership.mem A x₂
    x : E
    hxC : Membership.mem C x
    hx : Membership.mem (openSegment 𝕜 x₁ x₂) x
    hx₁B : Membership.mem B x₁
    hx₂B : Membership.mem B x₂
    ⊢ And (Membership.mem C x₁) (Membership.mem C x₂)
  -/
  exact hBC.2 hx₁B hx₂B hxC hx
  /-
    🎉 no goals
  -/


protected theorem IsExtreme.antisymm : AntiSymmetric (IsExtreme 𝕜 : Set E → Set E → Prop) :=
  fun _ _ hAB hBA ↦ Subset.antisymm hBA.1 hAB.1


instance : IsPartialOrder (Set E) (IsExtreme 𝕜) where
  refl := IsExtreme.refl 𝕜
  trans _ _ _ := IsExtreme.trans
  antisymm := IsExtreme.antisymm


theorem IsExtreme.inter (hAB : IsExtreme 𝕜 A B) (hAC : IsExtreme 𝕜 A C) :
    IsExtreme 𝕜 A (B ∩ C) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    A B C : Set E
    hAB : IsExtreme 𝕜 A B
    hAC : IsExtreme 𝕜 A C
    ⊢ IsExtreme 𝕜 A (Inter.inter B C)
  -/
  use Subset.trans inter_subset_left hAB.1
  /-
    case right
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    A B C : Set E
    hAB : IsExtreme 𝕜 A B
    hAC : IsExtreme 𝕜 A C
    ⊢ ∀ ⦃x₁ : E⦄, Membership.mem A x₁ → ∀ ⦃x₂ : E⦄, Membership.mem A x₂ → ∀ ⦃x : E …
  -/
  rintro x₁ hx₁A x₂ hx₂A x ⟨hxB, hxC⟩ hx
  /-
    case right.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    A B C : Set E
    hAB : IsExtreme 𝕜 A B
    hAC : IsExtreme 𝕜 A C
    x₁ : E
    hx₁A : Membership.mem A x₁
    x₂ : E
    hx₂A : Membership.mem A x₂
    x : E
    hxB : Membership.mem B x
    hxC : Membership.mem C x
    hx : Membership.mem (openSegment 𝕜 x₁ x₂) x
    ⊢ And (Membership.mem (Inter.inter B C) x₁) (Membership.mem (Inter.inter B C)  …
  -/
  obtain ⟨hx₁B, hx₂B⟩ := hAB.2 hx₁A hx₂A hxB hx
  /-
    case right.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    A B C : Set E
    hAB : IsExtreme 𝕜 A B
    hAC : IsExtreme 𝕜 A C
    x₁ : E
    hx₁A : Membership.mem A x₁
    x₂ : E
    hx₂A : Membership.mem A x₂
    x : E
    hxB : Membership.mem B x
    hxC : Membership.mem C x
    hx : Membership.mem (openSegment 𝕜 x₁ x₂) x
    hx₁B : Membership.mem B x₁
    hx₂B : Membership.mem B x₂
    ⊢ And (Membership.mem (Inter.inter B C) x₁) (Membership.mem (Inter.inter B C)  …
  -/
  obtain ⟨hx₁C, hx₂C⟩ := hAC.2 hx₁A hx₂A hxC hx
  /-
    case right.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    A B C : Set E
    hAB : IsExtreme 𝕜 A B
    hAC : IsExtreme 𝕜 A C
    x₁ : E
    hx₁A : Membership.mem A x₁
    x₂ : E
    hx₂A : Membership.mem A x₂
    x : E
    hxB : Membership.mem B x
    hxC : Membership.mem C x
    hx : Membership.mem (openSegment 𝕜 x₁ x₂) x
    hx₁B : Membership.mem B x₁
    hx₂B : Membership.mem B x₂
    hx₁C : Membership.mem C x₁
    hx₂C : Membership.mem C x₂
    ⊢ And (Membership.mem (Inter.inter B C) x₁) (Membership.mem (Inter.inter B C)  …
  -/
  exact ⟨⟨hx₁B, hx₁C⟩, hx₂B, hx₂C⟩
  /-
    🎉 no goals
  -/


protected theorem IsExtreme.mono (hAC : IsExtreme 𝕜 A C) (hBA : B ⊆ A) (hCB : C ⊆ B) :
    IsExtreme 𝕜 B C :=
  ⟨hCB, fun _ hx₁B _ hx₂B _ hxC hx ↦ hAC.2 (hBA hx₁B) (hBA hx₂B) hxC hx⟩


theorem isExtreme_iInter {ι : Sort*} [Nonempty ι] {F : ι → Set E}
    (hAF : ∀ i : ι, IsExtreme 𝕜 A (F i)) : IsExtreme 𝕜 A (⋂ i : ι, F i) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : SMul 𝕜 E
    A : Set E
    ι : Sort u_6
    inst✝ : Nonempty ι
    F : ι → Set E
    hAF : ∀ (i : ι), IsExtreme 𝕜 A (F i)
    ⊢ IsExtreme 𝕜 A (Set.iInter fun i => F i)
  -/
  obtain i := Classical.arbitrary ι
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : SMul 𝕜 E
    A : Set E
    ι : Sort u_6
    inst✝ : Nonempty ι
    F : ι → Set E
    hAF : ∀ (i : ι), IsExtreme 𝕜 A (F i)
    i : ι
    ⊢ IsExtreme 𝕜 A (Set.iInter fun i => F i)
  -/
  refine ⟨iInter_subset_of_subset i (hAF i).1, fun x₁ hx₁A x₂ hx₂A x hxF hx ↦ ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : SMul 𝕜 E
    A : Set E
    ι : Sort u_6
    inst✝ : Nonempty ι
    F : ι → Set E
    hAF : ∀ (i : ι), IsExtreme 𝕜 A (F i)
    i : ι
    x₁ : E
    hx₁A : Membership.mem A x₁
    x₂ : E
    hx₂A : Membership.mem A x₂
    x : E
    hxF : Membership.mem (Set.iInter fun i => F i) x
    hx : Membership.mem (openSegment 𝕜 x₁ x₂) x
    ⊢ And (Membership.mem (Set.iInter fun i => F i) x₁) (Membership.mem (Set.iInte …
  -/
  simp_rw [mem_iInter] at hxF ⊢
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : SMul 𝕜 E
    A : Set E
    ι : Sort u_6
    inst✝ : Nonempty ι
    F : ι → Set E
    hAF : ∀ (i : ι), IsExtreme 𝕜 A (F i)
    i : ι
    x₁ : E
    hx₁A : Membership.mem A x₁
    x₂ : E
    hx₂A : Membership.mem A x₂
    x : E
    hx : Membership.mem (openSegment 𝕜 x₁ x₂) x
    hxF : ∀ (i : ι), Membership.mem (F i) x
    ⊢ And (∀ (i : ι), Membership.mem (F i) x₁) (∀ (i : ι), Membership.mem (F i) x₂)
  -/
  have h := fun i ↦ (hAF i).2 hx₁A hx₂A (hxF i) hx
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : SMul 𝕜 E
    A : Set E
    ι : Sort u_6
    inst✝ : Nonempty ι
    F : ι → Set E
    hAF : ∀ (i : ι), IsExtreme 𝕜 A (F i)
    i : ι
    x₁ : E
    hx₁A : Membership.mem A x₁
    x₂ : E
    hx₂A : Membership.mem A x₂
    x : E
    hx : Membership.mem (openSegment 𝕜 x₁ x₂) x
    hxF : ∀ (i : ι), Membership.mem (F i) x
    h : ∀ (i : ι), And (Membership.mem (F i) x₁) (Membership.mem (F i) x₂)
    ⊢ And (∀ (i : ι), Membership.mem (F i) x₁) (∀ (i : ι), Membership.mem (F i) x₂)
  -/
  exact ⟨fun i ↦ (h i).1, fun i ↦ (h i).2⟩
  /-
    🎉 no goals
  -/


theorem isExtreme_biInter {F : Set (Set E)} (hF : F.Nonempty) (hA : ∀ B ∈ F, IsExtreme 𝕜 A B) :
    IsExtreme 𝕜 A (⋂ B ∈ F, B) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    A : Set E
    F : Set (Set E)
    hF : F.Nonempty
    hA : ∀ (B : Set E), Membership.mem F B → IsExtreme 𝕜 A B
    ⊢ IsExtreme 𝕜 A (Set.iInter fun B => Set.iInter fun h => B)
  -/
  haveI := hF.to_subtype
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    A : Set E
    F : Set (Set E)
    hF : F.Nonempty
    hA : ∀ (B : Set E), Membership.mem F B → IsExtreme 𝕜 A B
    this : Nonempty ↑F
    ⊢ IsExtreme 𝕜 A (Set.iInter fun B => Set.iInter fun h => B)
  -/
  simpa only [iInter_subtype] using isExtreme_iInter fun i : F ↦ hA _ i.2
  /-
    🎉 no goals
  -/


theorem isExtreme_sInter {F : Set (Set E)} (hF : F.Nonempty) (hAF : ∀ B ∈ F, IsExtreme 𝕜 A B) :
                               /-
                                 𝕜 : Type u_1
                                 E : Type u_2
                                 inst✝² : OrderedSemiring 𝕜
                                 inst✝¹ : AddCommMonoid E
                                 inst✝ : SMul 𝕜 E
                                 A : Set E
                                 F : Set (Set E)
                                 hF : F.Nonempty
                                 hAF : ∀ (B : Set E), Membership.mem F B → IsExtreme 𝕜 A B
                                 ⊢ IsExtreme 𝕜 A F.sInter
                               -/
    IsExtreme 𝕜 A (⋂₀ F) := by simpa [sInter_eq_biInter] using isExtreme_biInter hF hAF
                               /-
                                 🎉 no goals
                               -/


theorem mem_extremePoints : x ∈ A.extremePoints 𝕜 ↔
    x ∈ A ∧ ∀ᵉ (x₁ ∈ A) (x₂ ∈ A), x ∈ openSegment 𝕜 x₁ x₂ → x₁ = x ∧ x₂ = x :=
  Iff.rfl


/-- x is an extreme point to A iff {x} is an extreme set of A. -/
@[simp] lemma isExtreme_singleton : IsExtreme 𝕜 A {x} ↔ x ∈ A.extremePoints 𝕜 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    A : Set E
    x : E
    ⊢ Iff (IsExtreme 𝕜 A (Singleton.singleton x)) (Membership.mem (Set.extremePoin …
  -/
  refine ⟨fun hx ↦ ⟨singleton_subset_iff.1 hx.1, fun x₁ hx₁ x₂ hx₂ ↦ hx.2 hx₁ hx₂ rfl⟩, ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    A : Set E
    x : E
    ⊢ Membership.mem (Set.extremePoints 𝕜 A) x → IsExtreme 𝕜 A (Singleton.singleto …
  -/
  rintro ⟨hxA, hAx⟩
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    A : Set E
    x : E
    hxA : Membership.mem A x
    hAx : ∀ ⦃x₁ : E⦄, Membership.mem A x₁ → ∀ ⦃x₂ : E⦄, Membership.mem A x₂ → Memb …
    ⊢ IsExtreme 𝕜 A (Singleton.singleton x)
  -/
  use singleton_subset_iff.2 hxA
  /-
    case right
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    A : Set E
    x : E
    hxA : Membership.mem A x
    hAx : ∀ ⦃x₁ : E⦄, Membership.mem A x₁ → ∀ ⦃x₂ : E⦄, Membership.mem A x₂ → Memb …
    ⊢ ∀ ⦃x₁ : E⦄, Membership.mem A x₁ → ∀ ⦃x₂ : E⦄, Membership.mem A x₂ → ∀ ⦃x_1 : …
  -/
  rintro x₁ hx₁A x₂ hx₂A y (rfl : y = x)
  /-
    case right
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    A : Set E
    x₁ : E
    hx₁A : Membership.mem A x₁
    x₂ : E
    hx₂A : Membership.mem A x₂
    y : E
    hxA : Membership.mem A y
    hAx : ∀ ⦃x₁ : E⦄, Membership.mem A x₁ → ∀ ⦃x₂ : E⦄, Membership.mem A x₂ → Memb …
    ⊢ Membership.mem (openSegment 𝕜 x₁ x₂) y → And (Membership.mem (Singleton.sing …
  -/
  exact hAx hx₁A hx₂A
  /-
    🎉 no goals
  -/


alias ⟨IsExtreme.mem_extremePoints, _⟩ := isExtreme_singleton


theorem extremePoints_subset : A.extremePoints 𝕜 ⊆ A :=
  fun _ hx ↦ hx.1


@[simp]
theorem extremePoints_empty : (∅ : Set E).extremePoints 𝕜 = ∅ :=
  subset_empty_iff.1 extremePoints_subset


@[simp]
theorem extremePoints_singleton : ({x} : Set E).extremePoints 𝕜 = {x} :=
  extremePoints_subset.antisymm <|
    singleton_subset_iff.2 ⟨mem_singleton x, fun _ hx₁ _ hx₂ _ ↦ ⟨hx₁, hx₂⟩⟩


theorem inter_extremePoints_subset_extremePoints_of_subset (hBA : B ⊆ A) :
    B ∩ A.extremePoints 𝕜 ⊆ B.extremePoints 𝕜 :=
  fun _ ⟨hxB, hxA⟩ ↦ ⟨hxB, fun _ hx₁ _ hx₂ hx ↦ hxA.2 (hBA hx₁) (hBA hx₂) hx⟩


theorem IsExtreme.extremePoints_subset_extremePoints (hAB : IsExtreme 𝕜 A B) :
    B.extremePoints 𝕜 ⊆ A.extremePoints 𝕜 :=
             /-
               𝕜 : Type u_1
               E : Type u_2
               inst✝² : OrderedSemiring 𝕜
               inst✝¹ : AddCommMonoid E
               inst✝ : SMul 𝕜 E
               A B : Set E
               hAB : IsExtreme 𝕜 A B
               x✝ : E
               ⊢ Membership.mem (Set.extremePoints 𝕜 B) x✝ → Membership.mem (Set.extremePoint …
             -/
  fun _ ↦ by simpa only [← isExtreme_singleton] using hAB.trans
             /-
               🎉 no goals
             -/


theorem IsExtreme.extremePoints_eq (hAB : IsExtreme 𝕜 A B) :
    B.extremePoints 𝕜 = B ∩ A.extremePoints 𝕜 :=
  Subset.antisymm (fun _ hx ↦ ⟨hx.1, hAB.extremePoints_subset_extremePoints hx⟩)
    (inter_extremePoints_subset_extremePoints_of_subset hAB.1)


theorem IsExtreme.convex_diff (hA : Convex 𝕜 A) (hAB : IsExtreme 𝕜 A B) : Convex 𝕜 (A \ B) :=
  convex_iff_openSegment_subset.2 fun _ ⟨hx₁A, hx₁B⟩ _ ⟨hx₂A, _⟩ _ hx ↦
    ⟨hA.openSegment_subset hx₁A hx₂A hx, fun hxB ↦ hx₁B (hAB.2 hx₁A hx₂A hxB hx).1⟩


@[simp]
theorem extremePoints_prod (s : Set E) (t : Set F) :
    (s ×ˢ t).extremePoints 𝕜 = s.extremePoints 𝕜 ×ˢ t.extremePoints 𝕜 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    s : Set E
    t : Set F
    ⊢ Eq (Set.extremePoints 𝕜 (SProd.sprod s t)) (SProd.sprod (Set.extremePoints 𝕜 …
  -/
  ext
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    s : Set E
    t : Set F
    x✝ : Prod E F
    ⊢ Iff (Membership.mem (Set.extremePoints 𝕜 (SProd.sprod s t)) x✝) (Membership. …
  -/
  refine (and_congr_right fun hx ↦ ⟨fun h ↦ ?_, fun h ↦ ?_⟩).trans and_and_and_comm
  /-
    case h.refine_1
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    s : Set E
    t : Set F
    x✝ : Prod E F
    hx : Membership.mem (SProd.sprod s t) x✝
    h : ∀ ⦃x₁ : Prod E F⦄, Membership.mem (SProd.sprod s t) x₁ → ∀ ⦃x₂ : Prod E F⦄ …
    ⊢ And (∀ ⦃x₁ : E⦄, Membership.mem s x₁ → ∀ ⦃x₂ : E⦄, Membership.mem s x₂ → Mem …
  -/
  constructor
    /-
      case h.refine_1.left
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : AddCommGroup F
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 F
      s : Set E
      t : Set F
      x✝ : Prod E F
      hx : Membership.mem (SProd.sprod s t) x✝
      h : ∀ ⦃x₁ : Prod E F⦄, Membership.mem (SProd.sprod s t) x₁ → ∀ ⦃x₂ : Prod E F⦄ …
      ⊢ ∀ ⦃x₁ : E⦄, Membership.mem s x₁ → ∀ ⦃x₂ : E⦄, Membership.mem s x₂ → Membersh …
    -/
  · rintro x₁ hx₁ x₂ hx₂ hx_fst
    refine (h (mk_mem_prod hx₁ hx.2) (mk_mem_prod hx₂ hx.2) ?_).imp (congr_arg Prod.fst)
        (congr_arg Prod.fst)
    /-
      case h.refine_1.left
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : AddCommGroup F
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 F
      s : Set E
      t : Set F
      x✝ : Prod E F
      hx : Membership.mem (SProd.sprod s t) x✝
      h : ∀ ⦃x₁ : Prod E F⦄, Membership.mem (SProd.sprod s t) x₁ → ∀ ⦃x₂ : Prod E F⦄ …
      x₁ : E
      hx₁ : Membership.mem s x₁
      x₂ : E
      hx₂ : Membership.mem s x₂
      hx_fst : Membership.mem (openSegment 𝕜 x₁ x₂) x✝.1
      ⊢ Membership.mem (openSegment 𝕜 { fst := x₁, snd := x✝.2 } { fst := x₂, snd := …
    -/
    rw [← Prod.image_mk_openSegment_left]
    /-
      case h.refine_1.left
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : AddCommGroup F
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 F
      s : Set E
      t : Set F
      x✝ : Prod E F
      hx : Membership.mem (SProd.sprod s t) x✝
      h : ∀ ⦃x₁ : Prod E F⦄, Membership.mem (SProd.sprod s t) x₁ → ∀ ⦃x₂ : Prod E F⦄ …
      x₁ : E
      hx₁ : Membership.mem s x₁
      x₂ : E
      hx₂ : Membership.mem s x₂
      hx_fst : Membership.mem (openSegment 𝕜 x₁ x₂) x✝.1
      ⊢ Membership.mem (Set.image (fun x => { fst := x, snd := x✝.2 }) (openSegment  …
    -/
    exact ⟨_, hx_fst, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.refine_1.right
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : AddCommGroup F
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 F
      s : Set E
      t : Set F
      x✝ : Prod E F
      hx : Membership.mem (SProd.sprod s t) x✝
      h : ∀ ⦃x₁ : Prod E F⦄, Membership.mem (SProd.sprod s t) x₁ → ∀ ⦃x₂ : Prod E F⦄ …
      ⊢ ∀ ⦃x₁ : F⦄, Membership.mem t x₁ → ∀ ⦃x₂ : F⦄, Membership.mem t x₂ → Membersh …
    -/
  · rintro x₁ hx₁ x₂ hx₂ hx_snd
    refine (h (mk_mem_prod hx.1 hx₁) (mk_mem_prod hx.1 hx₂) ?_).imp (congr_arg Prod.snd)
        (congr_arg Prod.snd)
    /-
      case h.refine_1.right
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : AddCommGroup F
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 F
      s : Set E
      t : Set F
      x✝ : Prod E F
      hx : Membership.mem (SProd.sprod s t) x✝
      h : ∀ ⦃x₁ : Prod E F⦄, Membership.mem (SProd.sprod s t) x₁ → ∀ ⦃x₂ : Prod E F⦄ …
      x₁ : F
      hx₁ : Membership.mem t x₁
      x₂ : F
      hx₂ : Membership.mem t x₂
      hx_snd : Membership.mem (openSegment 𝕜 x₁ x₂) x✝.2
      ⊢ Membership.mem (openSegment 𝕜 { fst := x✝.1, snd := x₁ } { fst := x✝.1, snd  …
    -/
    rw [← Prod.image_mk_openSegment_right]
    /-
      case h.refine_1.right
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : AddCommGroup F
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 F
      s : Set E
      t : Set F
      x✝ : Prod E F
      hx : Membership.mem (SProd.sprod s t) x✝
      h : ∀ ⦃x₁ : Prod E F⦄, Membership.mem (SProd.sprod s t) x₁ → ∀ ⦃x₂ : Prod E F⦄ …
      x₁ : F
      hx₁ : Membership.mem t x₁
      x₂ : F
      hx₂ : Membership.mem t x₂
      hx_snd : Membership.mem (openSegment 𝕜 x₁ x₂) x✝.2
      ⊢ Membership.mem (Set.image (fun y => { fst := x✝.1, snd := y }) (openSegment  …
    -/
    exact ⟨_, hx_snd, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : AddCommGroup F
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 F
      s : Set E
      t : Set F
      x✝ : Prod E F
      hx : Membership.mem (SProd.sprod s t) x✝
      h : And (∀ ⦃x₁ : E⦄, Membership.mem s x₁ → ∀ ⦃x₂ : E⦄, Membership.mem s x₂ → M …
      ⊢ ∀ ⦃x₁ : Prod E F⦄, Membership.mem (SProd.sprod s t) x₁ → ∀ ⦃x₂ : Prod E F⦄,  …
    -/
  · rintro x₁ hx₁ x₂ hx₂ ⟨a, b, ha, hb, hab, hx'⟩
    /-
      case h.refine_2.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : AddCommGroup F
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 F
      s : Set E
      t : Set F
      x✝ : Prod E F
      hx : Membership.mem (SProd.sprod s t) x✝
      h : And (∀ ⦃x₁ : E⦄, Membership.mem s x₁ → ∀ ⦃x₂ : E⦄, Membership.mem s x₂ → M …
      x₁ : Prod E F
      hx₁ : Membership.mem (SProd.sprod s t) x₁
      x₂ : Prod E F
      hx₂ : Membership.mem (SProd.sprod s t) x₂
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      hx' : Eq (HAdd.hAdd (HSMul.hSMul a x₁) (HSMul.hSMul b x₂)) x✝
      ⊢ And (Eq x₁ x✝) (Eq x₂ x✝)
    -/
    simp_rw [Prod.ext_iff]
    exact and_and_and_comm.1
        ⟨h.1 hx₁.1 hx₂.1 ⟨a, b, ha, hb, hab, congr_arg Prod.fst hx'⟩,
          h.2 hx₁.2 hx₂.2 ⟨a, b, ha, hb, hab, congr_arg Prod.snd hx'⟩⟩


@[simp]
theorem extremePoints_pi (s : ∀ i, Set (π i)) :
    (univ.pi s).extremePoints 𝕜 = univ.pi fun i ↦ (s i).extremePoints 𝕜 := by
  classical
  ext x
  simp only [mem_extremePoints, mem_pi, mem_univ, true_imp_iff, @forall_and ι]
  refine and_congr_right fun hx ↦ ⟨fun h i ↦ ?_, fun h ↦ ?_⟩
  · rintro x₁ hx₁ x₂ hx₂ hi
    refine (h (update x i x₁) ?_ (update x i x₂) ?_ ?_).imp (fun h₁ ↦ by rw [← h₁, update_self])
        fun h₂ ↦ by rw [← h₂, update_self]
    iterate 2
      rintro j
      obtain rfl | hji := eq_or_ne j i
      · rwa [update_self]
      · rw [update_of_ne hji]
        exact hx _
    rw [← Pi.image_update_openSegment]
    exact ⟨_, hi, update_eq_self _ _⟩
  · rintro x₁ hx₁ x₂ hx₂ ⟨a, b, ha, hb, hab, hx'⟩
    simp_rw [funext_iff, ← forall_and]
    exact fun i ↦ h _ _ (hx₁ _) _ (hx₂ _) ⟨a, b, ha, hb, hab, congr_fun hx' _⟩


lemma image_extremePoints (f : L) (s : Set E) :
    f '' extremePoints 𝕜 s = extremePoints 𝕜 (f '' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    L : Type u_6
    inst✝⁶ : OrderedRing 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜 F
    inst✝¹ : EquivLike L E F
    inst✝ : LinearEquivClass L 𝕜 E F
    f : L
    s : Set E
    ⊢ Eq (Set.image (⇑f) (Set.extremePoints 𝕜 s)) (Set.extremePoints 𝕜 (Set.image  …
  -/
  ext b
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    L : Type u_6
    inst✝⁶ : OrderedRing 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜 F
    inst✝¹ : EquivLike L E F
    inst✝ : LinearEquivClass L 𝕜 E F
    f : L
    s : Set E
    b : F
    ⊢ Iff (Membership.mem (Set.image (⇑f) (Set.extremePoints 𝕜 s)) b) (Membership. …
  -/
  obtain ⟨a, rfl⟩ := EquivLike.surjective f b
  have : ∀ x y, f '' openSegment 𝕜 x y = openSegment 𝕜 (f x) (f y) :=
    image_openSegment _ (LinearMapClass.linearMap f).toAffineMap
  simp only [mem_extremePoints, (EquivLike.surjective f).forall,
    (EquivLike.injective f).mem_set_image, (EquivLike.injective f).eq_iff, ← this]


/-- A useful restatement using `segment`: `x` is an extreme point iff the only (closed) segments
that contain it are those with `x` as one of their endpoints. -/
theorem mem_extremePoints_iff_forall_segment : x ∈ A.extremePoints 𝕜 ↔
    x ∈ A ∧ ∀ᵉ (x₁ ∈ A) (x₂ ∈ A), x ∈ segment 𝕜 x₁ x₂ → x₁ = x ∨ x₂ = x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : LinearOrderedRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : DenselyOrdered 𝕜
    inst✝ : NoZeroSMulDivisors 𝕜 E
    A : Set E
    x : E
    ⊢ Iff (Membership.mem (Set.extremePoints 𝕜 A) x) (And (Membership.mem A x) (∀  …
  -/
  refine and_congr_right fun hxA ↦ forall₄_congr fun x₁ h₁ x₂ h₂ ↦ ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : LinearOrderedRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : DenselyOrdered 𝕜
    inst✝ : NoZeroSMulDivisors 𝕜 E
    A : Set E
    x : E
    hxA : Membership.mem A x
    x₁ : E
    h₁ : Membership.mem A x₁
    x₂ : E
    h₂ : Membership.mem A x₂
    ⊢ Iff (Membership.mem (openSegment 𝕜 x₁ x₂) x → And (Eq x₁ x) (Eq x₂ x)) (Memb …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : LinearOrderedRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : DenselyOrdered 𝕜
      inst✝ : NoZeroSMulDivisors 𝕜 E
      A : Set E
      x : E
      hxA : Membership.mem A x
      x₁ : E
      h₁ : Membership.mem A x₁
      x₂ : E
      h₂ : Membership.mem A x₂
      ⊢ (Membership.mem (openSegment 𝕜 x₁ x₂) x → And (Eq x₁ x) (Eq x₂ x)) → Members …
    -/
  · rw [← insert_endpoints_openSegment]
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : LinearOrderedRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : DenselyOrdered 𝕜
      inst✝ : NoZeroSMulDivisors 𝕜 E
      A : Set E
      x : E
      hxA : Membership.mem A x
      x₁ : E
      h₁ : Membership.mem A x₁
      x₂ : E
      h₂ : Membership.mem A x₂
      ⊢ (Membership.mem (openSegment 𝕜 x₁ x₂) x → And (Eq x₁ x) (Eq x₂ x)) → Members …
    -/
    rintro H (rfl | rfl | hx)
    /-
      case mp.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : LinearOrderedRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : DenselyOrdered 𝕜
      inst✝ : NoZeroSMulDivisors 𝕜 E
      A : Set E
      x : E
      hxA : Membership.mem A x
      x₂ : E
      h₂ : Membership.mem A x₂
      h₁ : Membership.mem A x
      H : Membership.mem (openSegment 𝕜 x x₂) x → And (Eq x x) (Eq x₂ x)
      ⊢ Or (Eq x x) (Eq x₂ x)
    -/
    exacts [Or.inl rfl, Or.inr rfl, Or.inl <| (H hx).1]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : LinearOrderedRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : DenselyOrdered 𝕜
      inst✝ : NoZeroSMulDivisors 𝕜 E
      A : Set E
      x : E
      hxA : Membership.mem A x
      x₁ : E
      h₁ : Membership.mem A x₁
      x₂ : E
      h₂ : Membership.mem A x₂
      ⊢ (Membership.mem (segment 𝕜 x₁ x₂) x → Or (Eq x₁ x) (Eq x₂ x)) → Membership.m …
    -/
  · intro H hx
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : LinearOrderedRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : DenselyOrdered 𝕜
      inst✝ : NoZeroSMulDivisors 𝕜 E
      A : Set E
      x : E
      hxA : Membership.mem A x
      x₁ : E
      h₁ : Membership.mem A x₁
      x₂ : E
      h₂ : Membership.mem A x₂
      H : Membership.mem (segment 𝕜 x₁ x₂) x → Or (Eq x₁ x) (Eq x₂ x)
      hx : Membership.mem (openSegment 𝕜 x₁ x₂) x
      ⊢ And (Eq x₁ x) (Eq x₂ x)
    -/
    rcases H (openSegment_subset_segment _ _ _ hx) with (rfl | rfl)
    /-
      case mpr.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : LinearOrderedRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : DenselyOrdered 𝕜
      inst✝ : NoZeroSMulDivisors 𝕜 E
      A : Set E
      x₁ : E
      h₁ : Membership.mem A x₁
      x₂ : E
      h₂ : Membership.mem A x₂
      hxA : Membership.mem A x₁
      H : Membership.mem (segment 𝕜 x₁ x₂) x₁ → Or (Eq x₁ x₁) (Eq x₂ x₁)
      hx : Membership.mem (openSegment 𝕜 x₁ x₂) x₁
      ⊢ And (Eq x₁ x₁) (Eq x₂ x₁)
    -/
    exacts [⟨rfl, (left_mem_openSegment_iff.1 hx).symm⟩, ⟨right_mem_openSegment_iff.1 hx, rfl⟩]
    /-
      🎉 no goals
    -/


theorem Convex.mem_extremePoints_iff_convex_diff (hA : Convex 𝕜 A) :
    x ∈ A.extremePoints 𝕜 ↔ x ∈ A ∧ Convex 𝕜 (A \ {x}) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : LinearOrderedRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : DenselyOrdered 𝕜
    inst✝ : NoZeroSMulDivisors 𝕜 E
    A : Set E
    x : E
    hA : Convex 𝕜 A
    ⊢ Iff (Membership.mem (Set.extremePoints 𝕜 A) x) (And (Membership.mem A x) (Co …
  -/
  use fun hx ↦ ⟨hx.1, (isExtreme_singleton.2 hx).convex_diff hA⟩
  /-
    case mpr
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : LinearOrderedRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : DenselyOrdered 𝕜
    inst✝ : NoZeroSMulDivisors 𝕜 E
    A : Set E
    x : E
    hA : Convex 𝕜 A
    ⊢ And (Membership.mem A x) (Convex 𝕜 (SDiff.sdiff A (Singleton.singleton x)))  …
  -/
  rintro ⟨hxA, hAx⟩
  /-
    case mpr.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : LinearOrderedRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : DenselyOrdered 𝕜
    inst✝ : NoZeroSMulDivisors 𝕜 E
    A : Set E
    x : E
    hA : Convex 𝕜 A
    hxA : Membership.mem A x
    hAx : Convex 𝕜 (SDiff.sdiff A (Singleton.singleton x))
    ⊢ Membership.mem (Set.extremePoints 𝕜 A) x
  -/
  refine mem_extremePoints_iff_forall_segment.2 ⟨hxA, fun x₁ hx₁ x₂ hx₂ hx ↦ ?_⟩
  /-
    case mpr.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : LinearOrderedRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : DenselyOrdered 𝕜
    inst✝ : NoZeroSMulDivisors 𝕜 E
    A : Set E
    x : E
    hA : Convex 𝕜 A
    hxA : Membership.mem A x
    hAx : Convex 𝕜 (SDiff.sdiff A (Singleton.singleton x))
    x₁ : E
    hx₁ : Membership.mem A x₁
    x₂ : E
    hx₂ : Membership.mem A x₂
    hx : Membership.mem (segment 𝕜 x₁ x₂) x
    ⊢ Or (Eq x₁ x) (Eq x₂ x)
  -/
  rw [convex_iff_segment_subset] at hAx
  /-
    case mpr.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : LinearOrderedRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : DenselyOrdered 𝕜
    inst✝ : NoZeroSMulDivisors 𝕜 E
    A : Set E
    x : E
    hA : Convex 𝕜 A
    hxA : Membership.mem A x
    hAx : ∀ ⦃x_1 : E⦄, Membership.mem (SDiff.sdiff A (Singleton.singleton x)) x_1  …
    x₁ : E
    hx₁ : Membership.mem A x₁
    x₂ : E
    hx₂ : Membership.mem A x₂
    hx : Membership.mem (segment 𝕜 x₁ x₂) x
    ⊢ Or (Eq x₁ x) (Eq x₂ x)
  -/
  by_contra! h
  exact (hAx ⟨hx₁, fun hx₁ ↦ h.1 (mem_singleton_iff.2 hx₁)⟩
      ⟨hx₂, fun hx₂ ↦ h.2 (mem_singleton_iff.2 hx₂)⟩ hx).2 rfl


theorem Convex.mem_extremePoints_iff_mem_diff_convexHull_diff (hA : Convex 𝕜 A) :
    x ∈ A.extremePoints 𝕜 ↔ x ∈ A \ convexHull 𝕜 (A \ {x}) := by
  rw [hA.mem_extremePoints_iff_convex_diff, hA.convex_remove_iff_not_mem_convexHull_remove,
    mem_diff]


theorem extremePoints_convexHull_subset : (convexHull 𝕜 A).extremePoints 𝕜 ⊆ A := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : LinearOrderedRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : DenselyOrdered 𝕜
    inst✝ : NoZeroSMulDivisors 𝕜 E
    A : Set E
    ⊢ HasSubset.Subset (Set.extremePoints 𝕜 ((convexHull 𝕜) A)) A
  -/
  rintro x hx
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : LinearOrderedRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : DenselyOrdered 𝕜
    inst✝ : NoZeroSMulDivisors 𝕜 E
    A : Set E
    x : E
    hx : Membership.mem (Set.extremePoints 𝕜 ((convexHull 𝕜) A)) x
    ⊢ Membership.mem A x
  -/
  rw [(convex_convexHull 𝕜 _).mem_extremePoints_iff_convex_diff] at hx
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : LinearOrderedRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : DenselyOrdered 𝕜
    inst✝ : NoZeroSMulDivisors 𝕜 E
    A : Set E
    x : E
    hx : And (Membership.mem ((convexHull 𝕜) A) x) (Convex 𝕜 (SDiff.sdiff ((convex …
    ⊢ Membership.mem A x
  -/
  by_contra h
  exact (convexHull_min (subset_diff.2 ⟨subset_convexHull 𝕜 _, disjoint_singleton_right.2 h⟩) hx.2
    hx.1).2 rfl


