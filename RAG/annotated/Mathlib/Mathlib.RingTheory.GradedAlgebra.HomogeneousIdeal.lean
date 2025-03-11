/-- An `I : Ideal A` is homogeneous if for every `r ∈ I`, all homogeneous components
  of `r` are in `I`. -/
def Ideal.IsHomogeneous : Prop :=
  ∀ (i : ι) ⦃r : A⦄, r ∈ I → (DirectSum.decompose 𝒜 r i : A) ∈ I


theorem Ideal.IsHomogeneous.mem_iff {I} (hI : Ideal.IsHomogeneous 𝒜 I) {x} :
    x ∈ I ↔ ∀ i, (decompose 𝒜 x i : A) ∈ I := by
  classical
  refine ⟨fun hx i ↦ hI i hx, fun hx ↦ ?_⟩
  rw [← DirectSum.sum_support_decompose 𝒜 x]
  exact Ideal.sum_mem _ (fun i _ ↦ hx i)


/-- For any `Semiring A`, we collect the homogeneous ideals of `A` into a type. -/
structure HomogeneousIdeal extends Submodule A A where
  is_homogeneous' : Ideal.IsHomogeneous 𝒜 toSubmodule


/-- Converting a homogeneous ideal to an ideal. -/
def HomogeneousIdeal.toIdeal (I : HomogeneousIdeal 𝒜) : Ideal A :=
  I.toSubmodule


theorem HomogeneousIdeal.isHomogeneous (I : HomogeneousIdeal 𝒜) : I.toIdeal.IsHomogeneous 𝒜 :=
  I.is_homogeneous'


theorem HomogeneousIdeal.toIdeal_injective :
    Function.Injective (HomogeneousIdeal.toIdeal : HomogeneousIdeal 𝒜 → Ideal A) :=
                                               /-
                                                 ι : Type u_1
                                                 σ : Type u_2
                                                 A : Type u_3
                                                 inst✝⁵ : Semiring A
                                                 inst✝⁴ : SetLike σ A
                                                 inst✝³ : AddSubmonoidClass σ A
                                                 𝒜 : ι → σ
                                                 inst✝² : DecidableEq ι
                                                 inst✝¹ : AddMonoid ι
                                                 inst✝ : GradedRing 𝒜
                                                 x✝¹ x✝ : HomogeneousIdeal 𝒜
                                                 x : Submodule A A
                                                 hx : Ideal.IsHomogeneous 𝒜 x
                                                 y : Submodule A A
                                                 hy : Ideal.IsHomogeneous 𝒜 y
                                                 h : Eq x y
                                                 ⊢ Eq { toSubmodule := x, is_homogeneous' := hx } { toSubmodule := y, is_homoge …
                                               -/
  fun ⟨x, hx⟩ ⟨y, hy⟩ => fun (h : x = y) => by simp [h]
                                               /-
                                                 🎉 no goals
                                               -/


instance HomogeneousIdeal.setLike : SetLike (HomogeneousIdeal 𝒜) A where
  coe I := I.toIdeal
  coe_injective' _ _ h := HomogeneousIdeal.toIdeal_injective <| SetLike.coe_injective h


@[ext]
theorem HomogeneousIdeal.ext {I J : HomogeneousIdeal 𝒜} (h : I.toIdeal = J.toIdeal) : I = J :=
  HomogeneousIdeal.toIdeal_injective h


theorem HomogeneousIdeal.ext' {I J : HomogeneousIdeal 𝒜} (h : ∀ i, ∀ x ∈ 𝒜 i, x ∈ I ↔ x ∈ J) :
    I = J := by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : SetLike σ A
    inst✝³ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : GradedRing 𝒜
    I J : HomogeneousIdeal 𝒜
    h : ∀ (i : ι) (x : A), Membership.mem (𝒜 i) x → Iff (Membership.mem I x) (Memb …
    ⊢ Eq I J
  -/
  ext
  /-
    case h.h
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : SetLike σ A
    inst✝³ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : GradedRing 𝒜
    I J : HomogeneousIdeal 𝒜
    h : ∀ (i : ι) (x : A), Membership.mem (𝒜 i) x → Iff (Membership.mem I x) (Memb …
    x✝ : A
    ⊢ Iff (Membership.mem I.toIdeal x✝) (Membership.mem J.toIdeal x✝)
  -/
  rw [I.isHomogeneous.mem_iff, J.isHomogeneous.mem_iff]
  /-
    case h.h
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : SetLike σ A
    inst✝³ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : GradedRing 𝒜
    I J : HomogeneousIdeal 𝒜
    h : ∀ (i : ι) (x : A), Membership.mem (𝒜 i) x → Iff (Membership.mem I x) (Memb …
    x✝ : A
    ⊢ Iff (∀ (i : ι), Membership.mem I.toIdeal ↑(((DirectSum.decompose 𝒜) x✝) i))  …
  -/
  apply forall_congr'
  /-
    case h.h.h
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : SetLike σ A
    inst✝³ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : GradedRing 𝒜
    I J : HomogeneousIdeal 𝒜
    h : ∀ (i : ι) (x : A), Membership.mem (𝒜 i) x → Iff (Membership.mem I x) (Memb …
    x✝ : A
    ⊢ ∀ (a : ι), Iff (Membership.mem I.toIdeal ↑(((DirectSum.decompose 𝒜) x✝) a))  …
  -/
  exact fun i ↦ h i _ (decompose 𝒜 _ i).2
  /-
    🎉 no goals
  -/


@[simp]
theorem HomogeneousIdeal.mem_iff {I : HomogeneousIdeal 𝒜} {x : A} : x ∈ I.toIdeal ↔ x ∈ I :=
  Iff.rfl


/-- For any `I : Ideal A`, not necessarily homogeneous, `I.homogeneousCore' 𝒜`
is the largest homogeneous ideal of `A` contained in `I`, as an ideal. -/
def Ideal.homogeneousCore' (I : Ideal A) : Ideal A :=
  Ideal.span ((↑) '' (((↑) : Subtype (Homogeneous 𝒜) → A) ⁻¹' I))


theorem Ideal.homogeneousCore'_mono : Monotone (Ideal.homogeneousCore' 𝒜) :=
  fun _ _ I_le_J => Ideal.span_mono <| Set.image_subset _ fun _ => @I_le_J _


theorem Ideal.homogeneousCore'_le : I.homogeneousCore' 𝒜 ≤ I :=
  Ideal.span_le.2 <| image_preimage_subset _ _


theorem Ideal.isHomogeneous_iff_forall_subset :
    I.IsHomogeneous 𝒜 ↔ ∀ i, (I : Set A) ⊆ GradedRing.proj 𝒜 i ⁻¹' I :=
  Iff.rfl


theorem Ideal.isHomogeneous_iff_subset_iInter :
    I.IsHomogeneous 𝒜 ↔ (I : Set A) ⊆ ⋂ i, GradedRing.proj 𝒜 i ⁻¹' ↑I :=
  subset_iInter_iff.symm


theorem Ideal.mul_homogeneous_element_mem_of_mem {I : Ideal A} (r x : A) (hx₁ : Homogeneous 𝒜 x)
    (hx₂ : x ∈ I) (j : ι) : GradedRing.proj 𝒜 j (r * x) ∈ I := by
  classical
  rw [← DirectSum.sum_support_decompose 𝒜 r, Finset.sum_mul, map_sum]
  apply Ideal.sum_mem
  intro k _
  obtain ⟨i, hi⟩ := hx₁
  have mem₁ : (DirectSum.decompose 𝒜 r k : A) * x ∈ 𝒜 (k + i) :=
    GradedMul.mul_mem (SetLike.coe_mem _) hi
  rw [GradedRing.proj_apply, DirectSum.decompose_of_mem 𝒜 mem₁, coe_of_apply]
  split_ifs
  · exact I.mul_mem_left _ hx₂
  · exact I.zero_mem


theorem Ideal.homogeneous_span (s : Set A) (h : ∀ x ∈ s, Homogeneous 𝒜 x) :
    (Ideal.span s).IsHomogeneous 𝒜 := by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : SetLike σ A
    inst✝³ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : GradedRing 𝒜
    s : Set A
    h : ∀ (x : A), Membership.mem s x → SetLike.Homogeneous 𝒜 x
    ⊢ Ideal.IsHomogeneous 𝒜 (Ideal.span s)
  -/
  rintro i r hr
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : SetLike σ A
    inst✝³ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : GradedRing 𝒜
    s : Set A
    h : ∀ (x : A), Membership.mem s x → SetLike.Homogeneous 𝒜 x
    i : ι
    r : A
    hr : Membership.mem (Ideal.span s) r
    ⊢ Membership.mem (Ideal.span s) ↑(((DirectSum.decompose 𝒜) r) i)
  -/
  rw [Ideal.span, Finsupp.span_eq_range_linearCombination] at hr
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : SetLike σ A
    inst✝³ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : GradedRing 𝒜
    s : Set A
    h : ∀ (x : A), Membership.mem s x → SetLike.Homogeneous 𝒜 x
    i : ι
    r : A
    hr : Membership.mem (LinearMap.range (Finsupp.linearCombination A Subtype.val) …
    ⊢ Membership.mem (Ideal.span s) ↑(((DirectSum.decompose 𝒜) r) i)
  -/
  rw [LinearMap.mem_range] at hr
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : SetLike σ A
    inst✝³ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : GradedRing 𝒜
    s : Set A
    h : ∀ (x : A), Membership.mem s x → SetLike.Homogeneous 𝒜 x
    i : ι
    r : A
    hr : Exists fun y => Eq ((Finsupp.linearCombination A Subtype.val) y) r
    ⊢ Membership.mem (Ideal.span s) ↑(((DirectSum.decompose 𝒜) r) i)
  -/
  obtain ⟨s, rfl⟩ := hr
  rw [Finsupp.linearCombination_apply, Finsupp.sum, decompose_sum, DFinsupp.finset_sum_apply,
    AddSubmonoidClass.coe_finset_sum]
  /-
    case intro
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : SetLike σ A
    inst✝³ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : GradedRing 𝒜
    s✝ : Set A
    h : ∀ (x : A), Membership.mem s✝ x → SetLike.Homogeneous 𝒜 x
    i : ι
    s : Finsupp (Subtype fun x => Membership.mem s✝ x) A
    ⊢ Membership.mem (Ideal.span s✝) (s.support.sum fun i_1 => ↑(((DirectSum.decom …
  -/
  refine Ideal.sum_mem _ ?_
  /-
    case intro
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : SetLike σ A
    inst✝³ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : GradedRing 𝒜
    s✝ : Set A
    h : ∀ (x : A), Membership.mem s✝ x → SetLike.Homogeneous 𝒜 x
    i : ι
    s : Finsupp (Subtype fun x => Membership.mem s✝ x) A
    ⊢ ∀ (c : Subtype fun x => Membership.mem s✝ x), Membership.mem s.support c → M …
  -/
  rintro z hz1
  /-
    case intro
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : SetLike σ A
    inst✝³ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : GradedRing 𝒜
    s✝ : Set A
    h : ∀ (x : A), Membership.mem s✝ x → SetLike.Homogeneous 𝒜 x
    i : ι
    s : Finsupp (Subtype fun x => Membership.mem s✝ x) A
    z : Subtype fun x => Membership.mem s✝ x
    hz1 : Membership.mem s.support z
    ⊢ Membership.mem (Ideal.span s✝) ↑(((DirectSum.decompose 𝒜) (HSMul.hSMul (s z) …
  -/
  rw [smul_eq_mul]
  /-
    case intro
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : SetLike σ A
    inst✝³ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : GradedRing 𝒜
    s✝ : Set A
    h : ∀ (x : A), Membership.mem s✝ x → SetLike.Homogeneous 𝒜 x
    i : ι
    s : Finsupp (Subtype fun x => Membership.mem s✝ x) A
    z : Subtype fun x => Membership.mem s✝ x
    hz1 : Membership.mem s.support z
    ⊢ Membership.mem (Ideal.span s✝) ↑(((DirectSum.decompose 𝒜) (HMul.hMul (s z) ↑ …
  -/
  refine Ideal.mul_homogeneous_element_mem_of_mem 𝒜 (s z) z ?_ ?_ i
    /-
      case intro.refine_1
      ι : Type u_1
      σ : Type u_2
      A : Type u_3
      inst✝⁵ : Semiring A
      inst✝⁴ : SetLike σ A
      inst✝³ : AddSubmonoidClass σ A
      𝒜 : ι → σ
      inst✝² : DecidableEq ι
      inst✝¹ : AddMonoid ι
      inst✝ : GradedRing 𝒜
      s✝ : Set A
      h : ∀ (x : A), Membership.mem s✝ x → SetLike.Homogeneous 𝒜 x
      i : ι
      s : Finsupp (Subtype fun x => Membership.mem s✝ x) A
      z : Subtype fun x => Membership.mem s✝ x
      hz1 : Membership.mem s.support z
      ⊢ SetLike.Homogeneous 𝒜 ↑z
    -/
  · rcases z with ⟨z, hz2⟩
    /-
      case intro.refine_1.mk
      ι : Type u_1
      σ : Type u_2
      A : Type u_3
      inst✝⁵ : Semiring A
      inst✝⁴ : SetLike σ A
      inst✝³ : AddSubmonoidClass σ A
      𝒜 : ι → σ
      inst✝² : DecidableEq ι
      inst✝¹ : AddMonoid ι
      inst✝ : GradedRing 𝒜
      s✝ : Set A
      h : ∀ (x : A), Membership.mem s✝ x → SetLike.Homogeneous 𝒜 x
      i : ι
      s : Finsupp (Subtype fun x => Membership.mem s✝ x) A
      z : A
      hz2 : Membership.mem s✝ z
      hz1 : Membership.mem s.support ⟨z, hz2⟩
      ⊢ SetLike.Homogeneous 𝒜 ↑⟨z, hz2⟩
    -/
    apply h _ hz2
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      ι : Type u_1
      σ : Type u_2
      A : Type u_3
      inst✝⁵ : Semiring A
      inst✝⁴ : SetLike σ A
      inst✝³ : AddSubmonoidClass σ A
      𝒜 : ι → σ
      inst✝² : DecidableEq ι
      inst✝¹ : AddMonoid ι
      inst✝ : GradedRing 𝒜
      s✝ : Set A
      h : ∀ (x : A), Membership.mem s✝ x → SetLike.Homogeneous 𝒜 x
      i : ι
      s : Finsupp (Subtype fun x => Membership.mem s✝ x) A
      z : Subtype fun x => Membership.mem s✝ x
      hz1 : Membership.mem s.support z
      ⊢ Membership.mem (Ideal.span s✝) ↑z
    -/
  · exact Ideal.subset_span z.2
    /-
      🎉 no goals
    -/


/-- For any `I : Ideal A`, not necessarily homogeneous, `I.homogeneousCore' 𝒜`
is the largest homogeneous ideal of `A` contained in `I`. -/
def Ideal.homogeneousCore : HomogeneousIdeal 𝒜 :=
  ⟨Ideal.homogeneousCore' 𝒜 I,
    Ideal.homogeneous_span _ _ fun _ h => by
      /-
        ι : Type u_1
        σ : Type u_2
        A : Type u_3
        inst✝⁵ : Semiring A
        inst✝⁴ : SetLike σ A
        inst✝³ : AddSubmonoidClass σ A
        𝒜 : ι → σ
        inst✝² : DecidableEq ι
        inst✝¹ : AddMonoid ι
        inst✝ : GradedRing 𝒜
        I : Ideal A
        x✝ : A
        h : Membership.mem (Set.image Subtype.val (Set.preimage Subtype.val ↑I)) x✝
        ⊢ SetLike.Homogeneous 𝒜 x✝
      -/
      have := Subtype.image_preimage_coe (setOf (Homogeneous 𝒜)) (I : Set A)
      /-
        ι : Type u_1
        σ : Type u_2
        A : Type u_3
        inst✝⁵ : Semiring A
        inst✝⁴ : SetLike σ A
        inst✝³ : AddSubmonoidClass σ A
        𝒜 : ι → σ
        inst✝² : DecidableEq ι
        inst✝¹ : AddMonoid ι
        inst✝ : GradedRing 𝒜
        I : Ideal A
        x✝ : A
        h : Membership.mem (Set.image Subtype.val (Set.preimage Subtype.val ↑I)) x✝
        this : Eq (Set.image Subtype.val (Set.preimage Subtype.val ↑I)) (Inter.inter ( …
        ⊢ SetLike.Homogeneous 𝒜 x✝
      -/
      exact (cast congr(_ ∈ $this) h).1⟩
      /-
        🎉 no goals
      -/


theorem Ideal.homogeneousCore_mono : Monotone (Ideal.homogeneousCore 𝒜) :=
  Ideal.homogeneousCore'_mono 𝒜


theorem Ideal.toIdeal_homogeneousCore_le : (I.homogeneousCore 𝒜).toIdeal ≤ I :=
  Ideal.homogeneousCore'_le 𝒜 I


theorem Ideal.mem_homogeneousCore_of_homogeneous_of_mem {x : A} (h : SetLike.Homogeneous 𝒜 x)
    (hmem : x ∈ I) : x ∈ I.homogeneousCore 𝒜 :=
  Ideal.subset_span ⟨⟨x, h⟩, hmem, rfl⟩


theorem Ideal.IsHomogeneous.toIdeal_homogeneousCore_eq_self (h : I.IsHomogeneous 𝒜) :
    (I.homogeneousCore 𝒜).toIdeal = I := by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : SetLike σ A
    inst✝³ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : GradedRing 𝒜
    I : Ideal A
    h : Ideal.IsHomogeneous 𝒜 I
    ⊢ Eq (Ideal.homogeneousCore 𝒜 I).toIdeal I
  -/
  apply le_antisymm (I.homogeneousCore'_le 𝒜) _
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : SetLike σ A
    inst✝³ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : GradedRing 𝒜
    I : Ideal A
    h : Ideal.IsHomogeneous 𝒜 I
    ⊢ LE.le I (Ideal.homogeneousCore' 𝒜 I)
  -/
  intro x hx
  classical
  rw [← DirectSum.sum_support_decompose 𝒜 x]
  exact Ideal.sum_mem _ fun j _ => Ideal.subset_span ⟨⟨_, homogeneous_coe _⟩, h _ hx, rfl⟩


@[simp]
theorem HomogeneousIdeal.toIdeal_homogeneousCore_eq_self (I : HomogeneousIdeal 𝒜) :
    I.toIdeal.homogeneousCore 𝒜 = I := by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : SetLike σ A
    inst✝³ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : GradedRing 𝒜
    I : HomogeneousIdeal 𝒜
    ⊢ Eq (Ideal.homogeneousCore 𝒜 I.toIdeal) I
  -/
  ext1
  /-
    case h
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : SetLike σ A
    inst✝³ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : GradedRing 𝒜
    I : HomogeneousIdeal 𝒜
    ⊢ Eq (Ideal.homogeneousCore 𝒜 I.toIdeal).toIdeal I.toIdeal
  -/
  convert Ideal.IsHomogeneous.toIdeal_homogeneousCore_eq_self I.isHomogeneous
  /-
    🎉 no goals
  -/


theorem Ideal.IsHomogeneous.iff_eq : I.IsHomogeneous 𝒜 ↔ (I.homogeneousCore 𝒜).toIdeal = I :=
  ⟨fun hI => hI.toIdeal_homogeneousCore_eq_self, fun hI => hI ▸ (Ideal.homogeneousCore 𝒜 I).2⟩


theorem Ideal.IsHomogeneous.iff_exists :
    I.IsHomogeneous 𝒜 ↔ ∃ S : Set (homogeneousSubmonoid 𝒜), I = Ideal.span ((↑) '' S) := by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : SetLike σ A
    inst✝³ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : GradedRing 𝒜
    I : Ideal A
    ⊢ Iff (Ideal.IsHomogeneous 𝒜 I) (Exists fun S => Eq I (Ideal.span (Set.image S …
  -/
  rw [Ideal.IsHomogeneous.iff_eq, eq_comm]
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : SetLike σ A
    inst✝³ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : GradedRing 𝒜
    I : Ideal A
    ⊢ Iff (Eq I (Ideal.homogeneousCore 𝒜 I).toIdeal) (Exists fun S => Eq I (Ideal. …
  -/
  exact ((Set.image_preimage.compose (Submodule.gi _ _).gc).exists_eq_l _).symm
  /-
    🎉 no goals
  -/


theorem bot : Ideal.IsHomogeneous 𝒜 ⊥ := fun i r hr => by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    i : ι
    r : A
    hr : Membership.mem Bot.bot r
    ⊢ Membership.mem Bot.bot ↑(((DirectSum.decompose 𝒜) r) i)
  -/
  simp only [Ideal.mem_bot] at hr
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    i : ι
    r : A
    hr : Eq r 0
    ⊢ Membership.mem Bot.bot ↑(((DirectSum.decompose 𝒜) r) i)
  -/
  rw [hr, decompose_zero, zero_apply]
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    i : ι
    r : A
    hr : Eq r 0
    ⊢ Membership.mem Bot.bot ↑0
  -/
  apply Ideal.zero_mem
  /-
    🎉 no goals
  -/


                                                         /-
                                                           ι : Type u_1
                                                           σ : Type u_2
                                                           A : Type u_3
                                                           inst✝⁵ : Semiring A
                                                           inst✝⁴ : DecidableEq ι
                                                           inst✝³ : AddMonoid ι
                                                           inst✝² : SetLike σ A
                                                           inst✝¹ : AddSubmonoidClass σ A
                                                           𝒜 : ι → σ
                                                           inst✝ : GradedRing 𝒜
                                                           i : ι
                                                           r : A
                                                           x✝ : Membership.mem Top.top r
                                                           ⊢ Membership.mem Top.top ↑(((DirectSum.decompose 𝒜) r) i)
                                                         -/
theorem top : Ideal.IsHomogeneous 𝒜 ⊤ := fun i r _ => by simp only [Submodule.mem_top]
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem inf {I J : Ideal A} (HI : I.IsHomogeneous 𝒜) (HJ : J.IsHomogeneous 𝒜) :
    (I ⊓ J).IsHomogeneous 𝒜 :=
  fun _ _ hr => ⟨HI _ hr.1, HJ _ hr.2⟩


theorem sup {I J : Ideal A} (HI : I.IsHomogeneous 𝒜) (HJ : J.IsHomogeneous 𝒜) :
    (I ⊔ J).IsHomogeneous 𝒜 := by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I J : Ideal A
    HI : Ideal.IsHomogeneous 𝒜 I
    HJ : Ideal.IsHomogeneous 𝒜 J
    ⊢ Ideal.IsHomogeneous 𝒜 (Max.max I J)
  -/
  rw [iff_exists] at HI HJ ⊢
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I J : Ideal A
    HI : Exists fun S => Eq I (Ideal.span (Set.image Subtype.val S))
    HJ : Exists fun S => Eq J (Ideal.span (Set.image Subtype.val S))
    ⊢ Exists fun S => Eq (Max.max I J) (Ideal.span (Set.image Subtype.val S))
  -/
  obtain ⟨⟨s₁, rfl⟩, ⟨s₂, rfl⟩⟩ := HI, HJ
  /-
    case intro.intro
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    s₁ s₂ : Set (Subtype fun x => Membership.mem (SetLike.homogeneousSubmonoid 𝒜) x)
    ⊢ Exists fun S => Eq (Max.max (Ideal.span (Set.image Subtype.val s₁)) (Ideal.s …
  -/
  refine ⟨s₁ ∪ s₂, ?_⟩
  /-
    case intro.intro
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    s₁ s₂ : Set (Subtype fun x => Membership.mem (SetLike.homogeneousSubmonoid 𝒜) x)
    ⊢ Eq (Max.max (Ideal.span (Set.image Subtype.val s₁)) (Ideal.span (Set.image S …
  -/
  rw [Set.image_union]
  /-
    case intro.intro
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    s₁ s₂ : Set (Subtype fun x => Membership.mem (SetLike.homogeneousSubmonoid 𝒜) x)
    ⊢ Eq (Max.max (Ideal.span (Set.image Subtype.val s₁)) (Ideal.span (Set.image S …
  -/
  exact (Submodule.span_union _ _).symm
  /-
    🎉 no goals
  -/


protected theorem iSup {κ : Sort*} {f : κ → Ideal A} (h : ∀ i, (f i).IsHomogeneous 𝒜) :
    (⨆ i, f i).IsHomogeneous 𝒜 := by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    κ : Sort u_4
    f : κ → Ideal A
    h : ∀ (i : κ), Ideal.IsHomogeneous 𝒜 (f i)
    ⊢ Ideal.IsHomogeneous 𝒜 (iSup fun i => f i)
  -/
  simp_rw [iff_exists] at h ⊢
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    κ : Sort u_4
    f : κ → Ideal A
    h : ∀ (i : κ), Exists fun S => Eq (f i) (Ideal.span (Set.image Subtype.val S))
    ⊢ Exists fun S => Eq (iSup fun i => f i) (Ideal.span (Set.image Subtype.val S))
  -/
  choose s hs using h
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    κ : Sort u_4
    f : κ → Ideal A
    s : κ → Set (Subtype fun x => Membership.mem (SetLike.homogeneousSubmonoid 𝒜) x)
    hs : ∀ (i : κ), Eq (f i) (Ideal.span (Set.image Subtype.val (s i)))
    ⊢ Exists fun S => Eq (iSup fun i => f i) (Ideal.span (Set.image Subtype.val S))
  -/
  refine ⟨⋃ i, s i, ?_⟩
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    κ : Sort u_4
    f : κ → Ideal A
    s : κ → Set (Subtype fun x => Membership.mem (SetLike.homogeneousSubmonoid 𝒜) x)
    hs : ∀ (i : κ), Eq (f i) (Ideal.span (Set.image Subtype.val (s i)))
    ⊢ Eq (iSup fun i => f i) (Ideal.span (Set.image Subtype.val (Set.iUnion fun i  …
  -/
  simp_rw [Set.image_iUnion, Ideal.span_iUnion]
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    κ : Sort u_4
    f : κ → Ideal A
    s : κ → Set (Subtype fun x => Membership.mem (SetLike.homogeneousSubmonoid 𝒜) x)
    hs : ∀ (i : κ), Eq (f i) (Ideal.span (Set.image Subtype.val (s i)))
    ⊢ Eq (iSup fun i => f i) (iSup fun i => Ideal.span (Set.image Subtype.val (s i …
  -/
  congr
  /-
    case e_s
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    κ : Sort u_4
    f : κ → Ideal A
    s : κ → Set (Subtype fun x => Membership.mem (SetLike.homogeneousSubmonoid 𝒜) x)
    hs : ∀ (i : κ), Eq (f i) (Ideal.span (Set.image Subtype.val (s i)))
    ⊢ Eq (fun i => f i) fun i => Ideal.span (Set.image Subtype.val (s i))
  -/
  exact funext hs
  /-
    🎉 no goals
  -/


protected theorem iInf {κ : Sort*} {f : κ → Ideal A} (h : ∀ i, (f i).IsHomogeneous 𝒜) :
    (⨅ i, f i).IsHomogeneous 𝒜 := by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    κ : Sort u_4
    f : κ → Ideal A
    h : ∀ (i : κ), Ideal.IsHomogeneous 𝒜 (f i)
    ⊢ Ideal.IsHomogeneous 𝒜 (iInf fun i => f i)
  -/
  intro i x hx
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    κ : Sort u_4
    f : κ → Ideal A
    h : ∀ (i : κ), Ideal.IsHomogeneous 𝒜 (f i)
    i : ι
    x : A
    hx : Membership.mem (iInf fun i => f i) x
    ⊢ Membership.mem (iInf fun i => f i) ↑(((DirectSum.decompose 𝒜) x) i)
  -/
  simp only [Ideal.mem_iInf] at hx ⊢
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    κ : Sort u_4
    f : κ → Ideal A
    h : ∀ (i : κ), Ideal.IsHomogeneous 𝒜 (f i)
    i : ι
    x : A
    hx : ∀ (i : κ), Membership.mem (f i) x
    ⊢ ∀ (i_1 : κ), Membership.mem (f i_1) ↑(((DirectSum.decompose 𝒜) x) i)
  -/
  exact fun j => h _ _ (hx j)
  /-
    🎉 no goals
  -/


theorem iSup₂ {κ : Sort*} {κ' : κ → Sort*} {f : ∀ i, κ' i → Ideal A}
    (h : ∀ i j, (f i j).IsHomogeneous 𝒜) : (⨆ (i) (j), f i j).IsHomogeneous 𝒜 :=
  IsHomogeneous.iSup fun i => IsHomogeneous.iSup <| h i


theorem iInf₂ {κ : Sort*} {κ' : κ → Sort*} {f : ∀ i, κ' i → Ideal A}
    (h : ∀ i j, (f i j).IsHomogeneous 𝒜) : (⨅ (i) (j), f i j).IsHomogeneous 𝒜 :=
  IsHomogeneous.iInf fun i => IsHomogeneous.iInf <| h i


theorem sSup {ℐ : Set (Ideal A)} (h : ∀ I ∈ ℐ, Ideal.IsHomogeneous 𝒜 I) :
    (sSup ℐ).IsHomogeneous 𝒜 := by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    ℐ : Set (Ideal A)
    h : ∀ (I : Ideal A), Membership.mem ℐ I → Ideal.IsHomogeneous 𝒜 I
    ⊢ Ideal.IsHomogeneous 𝒜 (SupSet.sSup ℐ)
  -/
  rw [sSup_eq_iSup]
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    ℐ : Set (Ideal A)
    h : ∀ (I : Ideal A), Membership.mem ℐ I → Ideal.IsHomogeneous 𝒜 I
    ⊢ Ideal.IsHomogeneous 𝒜 (iSup fun a => iSup fun h => a)
  -/
  exact iSup₂ h
  /-
    🎉 no goals
  -/


theorem sInf {ℐ : Set (Ideal A)} (h : ∀ I ∈ ℐ, Ideal.IsHomogeneous 𝒜 I) :
    (sInf ℐ).IsHomogeneous 𝒜 := by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    ℐ : Set (Ideal A)
    h : ∀ (I : Ideal A), Membership.mem ℐ I → Ideal.IsHomogeneous 𝒜 I
    ⊢ Ideal.IsHomogeneous 𝒜 (InfSet.sInf ℐ)
  -/
  rw [sInf_eq_iInf]
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    ℐ : Set (Ideal A)
    h : ∀ (I : Ideal A), Membership.mem ℐ I → Ideal.IsHomogeneous 𝒜 I
    ⊢ Ideal.IsHomogeneous 𝒜 (iInf fun a => iInf fun h => a)
  -/
  exact iInf₂ h
  /-
    🎉 no goals
  -/


instance : PartialOrder (HomogeneousIdeal 𝒜) :=
  SetLike.instPartialOrder


instance : Top (HomogeneousIdeal 𝒜) :=
  ⟨⟨⊤, Ideal.IsHomogeneous.top 𝒜⟩⟩


instance : Bot (HomogeneousIdeal 𝒜) :=
  ⟨⟨⊥, Ideal.IsHomogeneous.bot 𝒜⟩⟩


instance : Max (HomogeneousIdeal 𝒜) :=
  ⟨fun I J => ⟨_, I.isHomogeneous.sup J.isHomogeneous⟩⟩


instance : Min (HomogeneousIdeal 𝒜) :=
  ⟨fun I J => ⟨_, I.isHomogeneous.inf J.isHomogeneous⟩⟩


instance : SupSet (HomogeneousIdeal 𝒜) :=
  ⟨fun S => ⟨⨆ s ∈ S, toIdeal s, Ideal.IsHomogeneous.iSup₂ fun s _ => s.isHomogeneous⟩⟩


instance : InfSet (HomogeneousIdeal 𝒜) :=
  ⟨fun S => ⟨⨅ s ∈ S, toIdeal s, Ideal.IsHomogeneous.iInf₂ fun s _ => s.isHomogeneous⟩⟩


@[simp]
theorem coe_top : ((⊤ : HomogeneousIdeal 𝒜) : Set A) = univ :=
  rfl


@[simp]
theorem coe_bot : ((⊥ : HomogeneousIdeal 𝒜) : Set A) = 0 :=
  rfl


@[simp]
theorem coe_sup (I J : HomogeneousIdeal 𝒜) : ↑(I ⊔ J) = (I + J : Set A) :=
  Submodule.coe_sup _ _


@[simp]
theorem coe_inf (I J : HomogeneousIdeal 𝒜) : (↑(I ⊓ J) : Set A) = ↑I ∩ ↑J :=
  rfl


@[simp]
theorem toIdeal_top : (⊤ : HomogeneousIdeal 𝒜).toIdeal = (⊤ : Ideal A) :=
  rfl


@[simp]
theorem toIdeal_bot : (⊥ : HomogeneousIdeal 𝒜).toIdeal = (⊥ : Ideal A) :=
  rfl


@[simp]
theorem toIdeal_sup (I J : HomogeneousIdeal 𝒜) : (I ⊔ J).toIdeal = I.toIdeal ⊔ J.toIdeal :=
  rfl


@[simp]
theorem toIdeal_inf (I J : HomogeneousIdeal 𝒜) : (I ⊓ J).toIdeal = I.toIdeal ⊓ J.toIdeal :=
  rfl


@[simp]
theorem toIdeal_sSup (ℐ : Set (HomogeneousIdeal 𝒜)) : (sSup ℐ).toIdeal = ⨆ s ∈ ℐ, toIdeal s :=
  rfl


@[simp]
theorem toIdeal_sInf (ℐ : Set (HomogeneousIdeal 𝒜)) : (sInf ℐ).toIdeal = ⨅ s ∈ ℐ, toIdeal s :=
  rfl


@[simp]
theorem toIdeal_iSup {κ : Sort*} (s : κ → HomogeneousIdeal 𝒜) :
    (⨆ i, s i).toIdeal = ⨆ i, (s i).toIdeal := by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    κ : Sort u_4
    s : κ → HomogeneousIdeal 𝒜
    ⊢ Eq (iSup fun i => s i).toIdeal (iSup fun i => (s i).toIdeal)
  -/
  rw [iSup, toIdeal_sSup, iSup_range]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIdeal_iInf {κ : Sort*} (s : κ → HomogeneousIdeal 𝒜) :
    (⨅ i, s i).toIdeal = ⨅ i, (s i).toIdeal := by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    κ : Sort u_4
    s : κ → HomogeneousIdeal 𝒜
    ⊢ Eq (iInf fun i => s i).toIdeal (iInf fun i => (s i).toIdeal)
  -/
  rw [iInf, toIdeal_sInf, iInf_range]
  /-
    🎉 no goals
  -/


theorem toIdeal_iSup₂ {κ : Sort*} {κ' : κ → Sort*} (s : ∀ i, κ' i → HomogeneousIdeal 𝒜) :
    (⨆ (i) (j), s i j).toIdeal = ⨆ (i) (j), (s i j).toIdeal := by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    κ : Sort u_4
    κ' : κ → Sort u_5
    s : (i : κ) → κ' i → HomogeneousIdeal 𝒜
    ⊢ Eq (iSup fun i => iSup fun j => s i j).toIdeal (iSup fun i => iSup fun j =>  …
  -/
  simp_rw [toIdeal_iSup]
  /-
    🎉 no goals
  -/


theorem toIdeal_iInf₂ {κ : Sort*} {κ' : κ → Sort*} (s : ∀ i, κ' i → HomogeneousIdeal 𝒜) :
    (⨅ (i) (j), s i j).toIdeal = ⨅ (i) (j), (s i j).toIdeal := by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    κ : Sort u_4
    κ' : κ → Sort u_5
    s : (i : κ) → κ' i → HomogeneousIdeal 𝒜
    ⊢ Eq (iInf fun i => iInf fun j => s i j).toIdeal (iInf fun i => iInf fun j =>  …
  -/
  simp_rw [toIdeal_iInf]
  /-
    🎉 no goals
  -/


@[simp]
theorem eq_top_iff (I : HomogeneousIdeal 𝒜) : I = ⊤ ↔ I.toIdeal = ⊤ :=
  toIdeal_injective.eq_iff.symm


@[simp]
theorem eq_bot_iff (I : HomogeneousIdeal 𝒜) : I = ⊥ ↔ I.toIdeal = ⊥ :=
  toIdeal_injective.eq_iff.symm


instance completeLattice : CompleteLattice (HomogeneousIdeal 𝒜) :=
  toIdeal_injective.completeLattice _ toIdeal_sup toIdeal_inf toIdeal_sSup toIdeal_sInf toIdeal_top
    toIdeal_bot


instance : Add (HomogeneousIdeal 𝒜) :=
  ⟨(· ⊔ ·)⟩


@[simp]
theorem toIdeal_add (I J : HomogeneousIdeal 𝒜) : (I + J).toIdeal = I.toIdeal + J.toIdeal :=
  rfl


instance : Inhabited (HomogeneousIdeal 𝒜) where default := ⊥


theorem Ideal.IsHomogeneous.mul {I J : Ideal A} (HI : I.IsHomogeneous 𝒜) (HJ : J.IsHomogeneous 𝒜) :
    (I * J).IsHomogeneous 𝒜 := by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I J : Ideal A
    HI : Ideal.IsHomogeneous 𝒜 I
    HJ : Ideal.IsHomogeneous 𝒜 J
    ⊢ Ideal.IsHomogeneous 𝒜 (HMul.hMul I J)
  -/
  rw [Ideal.IsHomogeneous.iff_exists] at HI HJ ⊢
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I J : Ideal A
    HI : Exists fun S => Eq I (Ideal.span (Set.image Subtype.val S))
    HJ : Exists fun S => Eq J (Ideal.span (Set.image Subtype.val S))
    ⊢ Exists fun S => Eq (HMul.hMul I J) (Ideal.span (Set.image Subtype.val S))
  -/
  obtain ⟨⟨s₁, rfl⟩, ⟨s₂, rfl⟩⟩ := HI, HJ
  /-
    case intro.intro
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    s₁ s₂ : Set (Subtype fun x => Membership.mem (SetLike.homogeneousSubmonoid 𝒜) x)
    ⊢ Exists fun S => Eq (HMul.hMul (Ideal.span (Set.image Subtype.val s₁)) (Ideal …
  -/
  rw [Ideal.span_mul_span']
  /-
    case intro.intro
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    s₁ s₂ : Set (Subtype fun x => Membership.mem (SetLike.homogeneousSubmonoid 𝒜) x)
    ⊢ Exists fun S => Eq (Ideal.span (HMul.hMul (Set.image Subtype.val s₁) (Set.im …
  -/
  exact ⟨s₁ * s₂, congr_arg _ <| (Set.image_mul (homogeneousSubmonoid 𝒜).subtype).symm⟩
  /-
    🎉 no goals
  -/


instance : Mul (HomogeneousIdeal 𝒜) where
  mul I J := ⟨I.toIdeal * J.toIdeal, I.isHomogeneous.mul J.isHomogeneous⟩


@[simp]
theorem HomogeneousIdeal.toIdeal_mul (I J : HomogeneousIdeal 𝒜) :
    (I * J).toIdeal = I.toIdeal * J.toIdeal :=
  rfl


theorem Ideal.homogeneousCore.gc : GaloisConnection toIdeal (Ideal.homogeneousCore 𝒜) := fun I _ =>
  ⟨fun H => I.toIdeal_homogeneousCore_eq_self ▸ Ideal.homogeneousCore_mono 𝒜 H,
    fun H => le_trans H (Ideal.homogeneousCore'_le _ _)⟩


/-- `toIdeal : HomogeneousIdeal 𝒜 → Ideal A` and `Ideal.homogeneousCore 𝒜` forms a galois
coinsertion. -/
def Ideal.homogeneousCore.gi : GaloisCoinsertion toIdeal (Ideal.homogeneousCore 𝒜) where
  choice I HI :=
    ⟨I, le_antisymm (I.toIdeal_homogeneousCore_le 𝒜) HI ▸ HomogeneousIdeal.isHomogeneous _⟩
  gc := Ideal.homogeneousCore.gc 𝒜
  u_l_le _ := Ideal.homogeneousCore'_le _ _
  choice_eq I H := le_antisymm H (I.toIdeal_homogeneousCore_le _)


theorem Ideal.homogeneousCore_eq_sSup :
    I.homogeneousCore 𝒜 = sSup { J : HomogeneousIdeal 𝒜 | J.toIdeal ≤ I } :=
  Eq.symm <| IsLUB.sSup_eq <| (Ideal.homogeneousCore.gc 𝒜).isGreatest_u.isLUB


theorem Ideal.homogeneousCore'_eq_sSup :
    I.homogeneousCore' 𝒜 = sSup { J : Ideal A | J.IsHomogeneous 𝒜 ∧ J ≤ I } := by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    ⊢ Eq (Ideal.homogeneousCore' 𝒜 I) (SupSet.sSup (setOf fun J => And (Ideal.IsHo …
  -/
  refine (IsLUB.sSup_eq ?_).symm
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    ⊢ IsLUB (setOf fun J => And (Ideal.IsHomogeneous 𝒜 J) (LE.le J I)) (Ideal.homo …
  -/
  apply IsGreatest.isLUB
  /-
    case h
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    ⊢ IsGreatest (setOf fun J => And (Ideal.IsHomogeneous 𝒜 J) (LE.le J I)) (Ideal …
  -/
  have coe_mono : Monotone (toIdeal : HomogeneousIdeal 𝒜 → Ideal A) := fun x y => id
  /-
    case h
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    coe_mono : Monotone HomogeneousIdeal.toIdeal
    ⊢ IsGreatest (setOf fun J => And (Ideal.IsHomogeneous 𝒜 J) (LE.le J I)) (Ideal …
  -/
  convert coe_mono.map_isGreatest (Ideal.homogeneousCore.gc 𝒜).isGreatest_u using 1
  /-
    case h.e'_3
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    coe_mono : Monotone HomogeneousIdeal.toIdeal
    ⊢ Eq (setOf fun J => And (Ideal.IsHomogeneous 𝒜 J) (LE.le J I)) (Set.image Hom …
  -/
  ext x
  /-
    case h.e'_3.h
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    coe_mono : Monotone HomogeneousIdeal.toIdeal
    x : Ideal A
    ⊢ Iff (Membership.mem (setOf fun J => And (Ideal.IsHomogeneous 𝒜 J) (LE.le J I …
  -/
  rw [mem_image, mem_setOf_eq]
  /-
    case h.e'_3.h
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    coe_mono : Monotone HomogeneousIdeal.toIdeal
    x : Ideal A
    ⊢ Iff (And (Ideal.IsHomogeneous 𝒜 x) (LE.le x I)) (Exists fun x_1 => And (Memb …
  -/
  refine ⟨fun hI => ⟨⟨x, hI.1⟩, ⟨hI.2, rfl⟩⟩, ?_⟩
  /-
    case h.e'_3.h
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    coe_mono : Monotone HomogeneousIdeal.toIdeal
    x : Ideal A
    ⊢ (Exists fun x_1 => And (Membership.mem (setOf fun a => LE.le a.toIdeal I) x_ …
  -/
  rintro ⟨x, ⟨hx, rfl⟩⟩
  /-
    case h.e'_3.h.intro.intro
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    coe_mono : Monotone HomogeneousIdeal.toIdeal
    x : HomogeneousIdeal 𝒜
    hx : Membership.mem (setOf fun a => LE.le a.toIdeal I) x
    ⊢ And (Ideal.IsHomogeneous 𝒜 x.toIdeal) (LE.le x.toIdeal I)
  -/
  exact ⟨x.isHomogeneous, hx⟩
  /-
    🎉 no goals
  -/


/-- For any `I : Ideal A`, not necessarily homogeneous, `I.homogeneousHull 𝒜` is
the smallest homogeneous ideal containing `I`. -/
def Ideal.homogeneousHull : HomogeneousIdeal 𝒜 :=
  ⟨Ideal.span { r : A | ∃ (i : ι) (x : I), (DirectSum.decompose 𝒜 (x : A) i : A) = r }, by
    /-
      ι : Type u_1
      σ : Type u_2
      A : Type u_3
      inst✝⁵ : Semiring A
      inst✝⁴ : DecidableEq ι
      inst✝³ : AddMonoid ι
      inst✝² : SetLike σ A
      inst✝¹ : AddSubmonoidClass σ A
      𝒜 : ι → σ
      inst✝ : GradedRing 𝒜
      I : Ideal A
      ⊢ Ideal.IsHomogeneous 𝒜 (Ideal.span (setOf fun r => Exists fun i => Exists fun …
    -/
    refine Ideal.homogeneous_span _ _ fun x hx => ?_
    /-
      ι : Type u_1
      σ : Type u_2
      A : Type u_3
      inst✝⁵ : Semiring A
      inst✝⁴ : DecidableEq ι
      inst✝³ : AddMonoid ι
      inst✝² : SetLike σ A
      inst✝¹ : AddSubmonoidClass σ A
      𝒜 : ι → σ
      inst✝ : GradedRing 𝒜
      I : Ideal A
      x : A
      hx : Membership.mem (setOf fun r => Exists fun i => Exists fun x => Eq (↑(((Di …
      ⊢ SetLike.Homogeneous 𝒜 x
    -/
    obtain ⟨i, x, rfl⟩ := hx
    /-
      case intro.intro
      ι : Type u_1
      σ : Type u_2
      A : Type u_3
      inst✝⁵ : Semiring A
      inst✝⁴ : DecidableEq ι
      inst✝³ : AddMonoid ι
      inst✝² : SetLike σ A
      inst✝¹ : AddSubmonoidClass σ A
      𝒜 : ι → σ
      inst✝ : GradedRing 𝒜
      I : Ideal A
      i : ι
      x : Subtype fun x => Membership.mem I x
      ⊢ SetLike.Homogeneous 𝒜 ↑(((DirectSum.decompose 𝒜) ↑x) i)
    -/
    apply SetLike.homogeneous_coe⟩
    /-
      🎉 no goals
    -/


theorem Ideal.le_toIdeal_homogeneousHull : I ≤ (Ideal.homogeneousHull 𝒜 I).toIdeal := by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    ⊢ LE.le I (Ideal.homogeneousHull 𝒜 I).toIdeal
  -/
  intro r hr
  classical
  rw [← DirectSum.sum_support_decompose 𝒜 r]
  refine Ideal.sum_mem _ ?_
  intro j _
  apply Ideal.subset_span
  use j
  use ⟨r, hr⟩


theorem Ideal.homogeneousHull_mono : Monotone (Ideal.homogeneousHull 𝒜) := fun I J I_le_J => by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I J : Ideal A
    I_le_J : LE.le I J
    ⊢ LE.le (Ideal.homogeneousHull 𝒜 I) (Ideal.homogeneousHull 𝒜 J)
  -/
  apply Ideal.span_mono
  /-
    case a
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I J : Ideal A
    I_le_J : LE.le I J
    ⊢ HasSubset.Subset (setOf fun r => Exists fun i => Exists fun x => Eq (↑(((Dir …
  -/
  rintro r ⟨hr1, ⟨x, hx⟩, rfl⟩
  /-
    case a.intro.intro.mk
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I J : Ideal A
    I_le_J : LE.le I J
    hr1 : ι
    x : A
    hx : Membership.mem I x
    ⊢ Membership.mem (setOf fun r => Exists fun i => Exists fun x => Eq (↑(((Direc …
  -/
  exact ⟨hr1, ⟨⟨x, I_le_J hx⟩, rfl⟩⟩
  /-
    🎉 no goals
  -/


theorem Ideal.IsHomogeneous.toIdeal_homogeneousHull_eq_self (h : I.IsHomogeneous 𝒜) :
    (Ideal.homogeneousHull 𝒜 I).toIdeal = I := by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    h : Ideal.IsHomogeneous 𝒜 I
    ⊢ Eq (Ideal.homogeneousHull 𝒜 I).toIdeal I
  -/
  apply le_antisymm _ (Ideal.le_toIdeal_homogeneousHull _ _)
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    h : Ideal.IsHomogeneous 𝒜 I
    ⊢ LE.le (Ideal.homogeneousHull 𝒜 I).toIdeal I
  -/
  apply Ideal.span_le.2
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    h : Ideal.IsHomogeneous 𝒜 I
    ⊢ HasSubset.Subset (setOf fun r => Exists fun i => Exists fun x => Eq (↑(((Dir …
  -/
  rintro _ ⟨i, x, rfl⟩
  /-
    case intro.intro
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    h : Ideal.IsHomogeneous 𝒜 I
    i : ι
    x : Subtype fun x => Membership.mem I x
    ⊢ Membership.mem ↑I ↑(((DirectSum.decompose 𝒜) ↑x) i)
  -/
  exact h _ x.prop
  /-
    🎉 no goals
  -/


@[simp]
theorem HomogeneousIdeal.homogeneousHull_toIdeal_eq_self (I : HomogeneousIdeal 𝒜) :
    I.toIdeal.homogeneousHull 𝒜 = I :=
  HomogeneousIdeal.toIdeal_injective <| I.isHomogeneous.toIdeal_homogeneousHull_eq_self


theorem Ideal.toIdeal_homogeneousHull_eq_iSup :
    (I.homogeneousHull 𝒜).toIdeal = ⨆ i, Ideal.span (GradedRing.proj 𝒜 i '' I) := by
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    ⊢ Eq (Ideal.homogeneousHull 𝒜 I).toIdeal (iSup fun i => Ideal.span (Set.image  …
  -/
  rw [← Ideal.span_iUnion]
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    ⊢ Eq (Ideal.homogeneousHull 𝒜 I).toIdeal (Ideal.span (Set.iUnion fun i => Set. …
  -/
  apply congr_arg Ideal.span _
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    ⊢ Eq (setOf fun r => Exists fun i => Exists fun x => Eq (↑(((DirectSum.decompo …
  -/
  ext1
  simp only [Set.mem_iUnion, Set.mem_image, mem_setOf_eq, GradedRing.proj_apply, SetLike.exists,
    exists_prop, Subtype.coe_mk, SetLike.mem_coe]


theorem Ideal.homogeneousHull_eq_iSup :
    I.homogeneousHull 𝒜 =
      ⨆ i, ⟨Ideal.span (GradedRing.proj 𝒜 i '' I), Ideal.homogeneous_span 𝒜 _ (by
        /-
          ι : Type u_1
          σ : Type u_2
          A : Type u_3
          inst✝⁵ : Semiring A
          inst✝⁴ : DecidableEq ι
          inst✝³ : AddMonoid ι
          inst✝² : SetLike σ A
          inst✝¹ : AddSubmonoidClass σ A
          𝒜 : ι → σ
          inst✝ : GradedRing 𝒜
          I : Ideal A
          i : ι
          ⊢ ∀ (x : A), Membership.mem (Set.image ⇑(GradedRing.proj 𝒜 i) ↑I) x → SetLike. …
        -/
        rintro _ ⟨x, -, rfl⟩
        /-
          case intro.intro
          ι : Type u_1
          σ : Type u_2
          A : Type u_3
          inst✝⁵ : Semiring A
          inst✝⁴ : DecidableEq ι
          inst✝³ : AddMonoid ι
          inst✝² : SetLike σ A
          inst✝¹ : AddSubmonoidClass σ A
          𝒜 : ι → σ
          inst✝ : GradedRing 𝒜
          I : Ideal A
          i : ι
          x : A
          ⊢ SetLike.Homogeneous 𝒜 ((GradedRing.proj 𝒜 i) x)
        -/
        apply SetLike.homogeneous_coe)⟩ := by
        /-
          🎉 no goals
        -/
  /-
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    ⊢ Eq (Ideal.homogeneousHull 𝒜 I) (iSup fun i => { toSubmodule := Ideal.span (S …
  -/
  ext1
  /-
    case h
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    ⊢ Eq (Ideal.homogeneousHull 𝒜 I).toIdeal (iSup fun i => { toSubmodule := Ideal …
  -/
  rw [Ideal.toIdeal_homogeneousHull_eq_iSup, toIdeal_iSup]
  /-
    case h
    ι : Type u_1
    σ : Type u_2
    A : Type u_3
    inst✝⁵ : Semiring A
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddMonoid ι
    inst✝² : SetLike σ A
    inst✝¹ : AddSubmonoidClass σ A
    𝒜 : ι → σ
    inst✝ : GradedRing 𝒜
    I : Ideal A
    ⊢ Eq (iSup fun i => Ideal.span (Set.image ⇑(GradedRing.proj 𝒜 i) ↑I)) (iSup fu …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Ideal.homogeneousHull.gc : GaloisConnection (Ideal.homogeneousHull 𝒜) toIdeal := fun _ J =>
  ⟨le_trans (Ideal.le_toIdeal_homogeneousHull _ _),
    fun H => J.homogeneousHull_toIdeal_eq_self ▸ Ideal.homogeneousHull_mono 𝒜 H⟩


/-- `Ideal.homogeneousHull 𝒜` and `toIdeal : HomogeneousIdeal 𝒜 → Ideal A` form a galois
insertion. -/
def Ideal.homogeneousHull.gi : GaloisInsertion (Ideal.homogeneousHull 𝒜) toIdeal where
  choice I H := ⟨I, le_antisymm H (I.le_toIdeal_homogeneousHull 𝒜) ▸ isHomogeneous _⟩
  gc := Ideal.homogeneousHull.gc 𝒜
  le_l_u _ := Ideal.le_toIdeal_homogeneousHull _ _
  choice_eq I H := le_antisymm (I.le_toIdeal_homogeneousHull 𝒜) H


theorem Ideal.homogeneousHull_eq_sInf (I : Ideal A) :
    Ideal.homogeneousHull 𝒜 I = sInf { J : HomogeneousIdeal 𝒜 | I ≤ J.toIdeal } :=
  Eq.symm <| IsGLB.sInf_eq <| (Ideal.homogeneousHull.gc 𝒜).isLeast_l.isGLB


/-- For a graded ring `⨁ᵢ 𝒜ᵢ` graded by a `CanonicallyOrderedAddCommMonoid ι`, the irrelevant ideal
refers to `⨁_{i>0} 𝒜ᵢ`, or equivalently `{a | a₀ = 0}`. This definition is used in `Proj`
construction where `ι` is always `ℕ` so the irrelevant ideal is simply elements with `0` as
0-th coordinate.

# Future work
Here in the definition, `ι` is assumed to be `CanonicallyOrderedAddCommMonoid`. However, the notion
of irrelevant ideal makes sense in a more general setting by defining it as the ideal of elements
with `0` as i-th coordinate for all `i ≤ 0`, i.e. `{a | ∀ (i : ι), i ≤ 0 → aᵢ = 0}`.
-/
def HomogeneousIdeal.irrelevant : HomogeneousIdeal 𝒜 :=
  ⟨RingHom.ker (GradedRing.projZeroRingHom 𝒜), fun i r (hr : (decompose 𝒜 r 0 : A) = 0) => by
    /-
      ι : Type u_1
      σ : Type u_2
      A : Type u_3
      inst✝⁵ : Semiring A
      inst✝⁴ : DecidableEq ι
      inst✝³ : CanonicallyOrderedAddCommMonoid ι
      inst✝² : SetLike σ A
      inst✝¹ : AddSubmonoidClass σ A
      𝒜 : ι → σ
      inst✝ : GradedRing 𝒜
      i : ι
      r : A
      hr : Eq (↑(((DirectSum.decompose 𝒜) r) 0)) 0
      ⊢ Membership.mem (RingHom.ker (GradedRing.projZeroRingHom 𝒜)) ↑(((DirectSum.de …
    -/
    change (decompose 𝒜 (decompose 𝒜 r _ : A) 0 : A) = 0
    /-
      ι : Type u_1
      σ : Type u_2
      A : Type u_3
      inst✝⁵ : Semiring A
      inst✝⁴ : DecidableEq ι
      inst✝³ : CanonicallyOrderedAddCommMonoid ι
      inst✝² : SetLike σ A
      inst✝¹ : AddSubmonoidClass σ A
      𝒜 : ι → σ
      inst✝ : GradedRing 𝒜
      i : ι
      r : A
      hr : Eq (↑(((DirectSum.decompose 𝒜) r) 0)) 0
      ⊢ Eq (↑(((DirectSum.decompose 𝒜) ↑(((DirectSum.decompose 𝒜) r) i)) 0)) 0
    -/
    by_cases h : i = 0
      /-
        case pos
        ι : Type u_1
        σ : Type u_2
        A : Type u_3
        inst✝⁵ : Semiring A
        inst✝⁴ : DecidableEq ι
        inst✝³ : CanonicallyOrderedAddCommMonoid ι
        inst✝² : SetLike σ A
        inst✝¹ : AddSubmonoidClass σ A
        𝒜 : ι → σ
        inst✝ : GradedRing 𝒜
        i : ι
        r : A
        hr : Eq (↑(((DirectSum.decompose 𝒜) r) 0)) 0
        h : Eq i 0
        ⊢ Eq (↑(((DirectSum.decompose 𝒜) ↑(((DirectSum.decompose 𝒜) r) i)) 0)) 0
      -/
    · rw [h, hr, decompose_zero, zero_apply, ZeroMemClass.coe_zero]
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u_1
        σ : Type u_2
        A : Type u_3
        inst✝⁵ : Semiring A
        inst✝⁴ : DecidableEq ι
        inst✝³ : CanonicallyOrderedAddCommMonoid ι
        inst✝² : SetLike σ A
        inst✝¹ : AddSubmonoidClass σ A
        𝒜 : ι → σ
        inst✝ : GradedRing 𝒜
        i : ι
        r : A
        hr : Eq (↑(((DirectSum.decompose 𝒜) r) 0)) 0
        h : Not (Eq i 0)
        ⊢ Eq (↑(((DirectSum.decompose 𝒜) ↑(((DirectSum.decompose 𝒜) r) i)) 0)) 0
      -/
    · rw [decompose_of_mem_ne 𝒜 (SetLike.coe_mem _) h]⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem HomogeneousIdeal.mem_irrelevant_iff (a : A) :
    a ∈ HomogeneousIdeal.irrelevant 𝒜 ↔ proj 𝒜 0 a = 0 :=
  Iff.rfl


@[simp]
theorem HomogeneousIdeal.toIdeal_irrelevant :
    (HomogeneousIdeal.irrelevant 𝒜).toIdeal = RingHom.ker (GradedRing.projZeroRingHom 𝒜) :=
  rfl


