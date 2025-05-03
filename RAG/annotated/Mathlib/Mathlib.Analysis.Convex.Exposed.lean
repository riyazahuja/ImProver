/-- A set `B` is exposed with respect to `A` iff it maximizes some functional over `A` (and contains
all points maximizing it). Written `IsExposed 𝕜 A B`. -/
def IsExposed (A B : Set E) : Prop :=
  B.Nonempty → ∃ l : E →L[𝕜] 𝕜, B = { x ∈ A | ∀ y ∈ A, l y ≤ l x }


/-- A useful way to build exposed sets from intersecting `A` with half-spaces (modelled by an
inequality with a functional). -/
def ContinuousLinearMap.toExposed (l : E →L[𝕜] 𝕜) (A : Set E) : Set E :=
  { x ∈ A | ∀ y ∈ A, l y ≤ l x }


theorem ContinuousLinearMap.toExposed.isExposed : IsExposed 𝕜 A (l.toExposed A) := fun _ => ⟨l, rfl⟩


theorem isExposed_empty : IsExposed 𝕜 A ∅ := fun ⟨_, hx⟩ => by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : OrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A : Set E
    x✝ : EmptyCollection.emptyCollection.Nonempty
    w✝ : E
    hx : Membership.mem EmptyCollection.emptyCollection w✝
    ⊢ Exists fun l => Eq EmptyCollection.emptyCollection (setOf fun x => And (Memb …
  -/
  exfalso
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : OrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A : Set E
    x✝ : EmptyCollection.emptyCollection.Nonempty
    w✝ : E
    hx : Membership.mem EmptyCollection.emptyCollection w✝
    ⊢ False
  -/
  exact hx
  /-
    🎉 no goals
  -/


protected theorem subset (hAB : IsExposed 𝕜 A B) : B ⊆ A := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : OrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A B : Set E
    hAB : IsExposed 𝕜 A B
    ⊢ HasSubset.Subset B A
  -/
  rintro x hx
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : OrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A B : Set E
    hAB : IsExposed 𝕜 A B
    x : E
    hx : Membership.mem B x
    ⊢ Membership.mem A x
  -/
  obtain ⟨_, rfl⟩ := hAB ⟨x, hx⟩
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : OrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A : Set E
    x : E
    w✝ : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hAB : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Membe …
    hx : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Membe …
    ⊢ Membership.mem A x
  -/
  exact hx.1
  /-
    🎉 no goals
  -/


@[refl]
protected theorem refl (A : Set E) : IsExposed 𝕜 A A := fun ⟨_, _⟩ =>
  ⟨0, Subset.antisymm (fun _ hx => ⟨hx, fun _ _ => le_refl 0⟩) fun _ hx => hx.1⟩


protected theorem antisymm (hB : IsExposed 𝕜 A B) (hA : IsExposed 𝕜 B A) : A = B :=
  hA.subset.antisymm hB.subset


protected theorem mono (hC : IsExposed 𝕜 A C) (hBA : B ⊆ A) (hCB : C ⊆ B) : IsExposed 𝕜 B C := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : OrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A B C : Set E
    hC : IsExposed 𝕜 A C
    hBA : HasSubset.Subset B A
    hCB : HasSubset.Subset C B
    ⊢ IsExposed 𝕜 B C
  -/
  rintro ⟨w, hw⟩
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : OrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A B C : Set E
    hC : IsExposed 𝕜 A C
    hBA : HasSubset.Subset B A
    hCB : HasSubset.Subset C B
    w : E
    hw : Membership.mem C w
    ⊢ Exists fun l => Eq C (setOf fun x => And (Membership.mem B x) (∀ (y : E), Me …
  -/
  obtain ⟨l, rfl⟩ := hC ⟨w, hw⟩
  exact ⟨l, Subset.antisymm (fun x hx => ⟨hCB hx, fun y hy => hx.2 y (hBA hy)⟩) fun x hx =>
    ⟨hBA hx.1, fun y hy => (hw.2 y hy).trans (hx.2 w (hCB hw))⟩⟩


/-- If `B` is a nonempty exposed subset of `A`, then `B` is the intersection of `A` with some closed
half-space. The converse is *not* true. It would require that the corresponding open half-space
doesn't intersect `A`. -/
theorem eq_inter_halfSpace' {A B : Set E} (hAB : IsExposed 𝕜 A B) (hB : B.Nonempty) :
    ∃ l : E →L[𝕜] 𝕜, ∃ a, B = { x ∈ A | a ≤ l x } := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : OrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A B : Set E
    hAB : IsExposed 𝕜 A B
    hB : B.Nonempty
    ⊢ Exists fun l => Exists fun a => Eq B (setOf fun x => And (Membership.mem A x …
  -/
  obtain ⟨l, rfl⟩ := hAB hB
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : OrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A : Set E
    l : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hAB : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Membe …
    hB : (setOf fun x => And (Membership.mem A x) (∀ (y : E), Membership.mem A y → …
    ⊢ Exists fun l_1 => Exists fun a => Eq (setOf fun x => And (Membership.mem A x …
  -/
  obtain ⟨w, hw⟩ := hB
  exact ⟨l, l w, Subset.antisymm (fun x hx => ⟨hx.1, hx.2 w hw.1⟩) fun x hx =>
    ⟨hx.1, fun y hy => (hw.2 y hy).trans hx.2⟩⟩

@[deprecated (since := "2024-11-12")] alias eq_inter_halfspace' := eq_inter_halfSpace'


/-- For nontrivial `𝕜`, if `B` is an exposed subset of `A`, then `B` is the intersection of `A` with
some closed half-space. The converse is *not* true. It would require that the corresponding open
half-space doesn't intersect `A`. -/
theorem eq_inter_halfSpace [Nontrivial 𝕜] {A B : Set E} (hAB : IsExposed 𝕜 A B) :
    ∃ l : E →L[𝕜] 𝕜, ∃ a, B = { x ∈ A | a ≤ l x } := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : TopologicalSpace 𝕜
    inst✝⁴ : OrderedRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : Nontrivial 𝕜
    A B : Set E
    hAB : IsExposed 𝕜 A B
    ⊢ Exists fun l => Exists fun a => Eq B (setOf fun x => And (Membership.mem A x …
  -/
  obtain rfl | hB := B.eq_empty_or_nonempty
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : TopologicalSpace 𝕜
      inst✝⁴ : OrderedRing 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : Nontrivial 𝕜
      A : Set E
      hAB : IsExposed 𝕜 A EmptyCollection.emptyCollection
      ⊢ Exists fun l => Exists fun a => Eq EmptyCollection.emptyCollection (setOf fu …
    -/
  · refine ⟨0, 1, ?_⟩
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : TopologicalSpace 𝕜
      inst✝⁴ : OrderedRing 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : Nontrivial 𝕜
      A : Set E
      hAB : IsExposed 𝕜 A EmptyCollection.emptyCollection
      ⊢ Eq EmptyCollection.emptyCollection (setOf fun x => And (Membership.mem A x)  …
    -/
    rw [eq_comm, eq_empty_iff_forall_not_mem]
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : TopologicalSpace 𝕜
      inst✝⁴ : OrderedRing 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : Nontrivial 𝕜
      A : Set E
      hAB : IsExposed 𝕜 A EmptyCollection.emptyCollection
      ⊢ ∀ (x : E), Not (Membership.mem (setOf fun x => And (Membership.mem A x) (LE. …
    -/
    rintro x ⟨-, h⟩
    /-
      case inl.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : TopologicalSpace 𝕜
      inst✝⁴ : OrderedRing 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : Nontrivial 𝕜
      A : Set E
      hAB : IsExposed 𝕜 A EmptyCollection.emptyCollection
      x : E
      h : LE.le 1 (0 x)
      ⊢ False
    -/
    rw [ContinuousLinearMap.zero_apply] at h
    /-
      case inl.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : TopologicalSpace 𝕜
      inst✝⁴ : OrderedRing 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : Nontrivial 𝕜
      A : Set E
      hAB : IsExposed 𝕜 A EmptyCollection.emptyCollection
      x : E
      h : LE.le 1 0
      ⊢ False
    -/
    have : ¬(1 : 𝕜) ≤ 0 := not_le_of_lt zero_lt_one
    /-
      case inl.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : TopologicalSpace 𝕜
      inst✝⁴ : OrderedRing 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : Nontrivial 𝕜
      A : Set E
      hAB : IsExposed 𝕜 A EmptyCollection.emptyCollection
      x : E
      h : LE.le 1 0
      this : Not (LE.le 1 0)
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : TopologicalSpace 𝕜
    inst✝⁴ : OrderedRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : Nontrivial 𝕜
    A B : Set E
    hAB : IsExposed 𝕜 A B
    hB : B.Nonempty
    ⊢ Exists fun l => Exists fun a => Eq B (setOf fun x => And (Membership.mem A x …
  -/
  exact hAB.eq_inter_halfSpace' hB
  /-
    🎉 no goals
  -/

@[deprecated (since := "2024-11-12")] alias eq_inter_halfspace := eq_inter_halfSpace


protected theorem inter [ContinuousAdd 𝕜] {A B C : Set E} (hB : IsExposed 𝕜 A B)
    (hC : IsExposed 𝕜 A C) : IsExposed 𝕜 A (B ∩ C) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : TopologicalSpace 𝕜
    inst✝⁴ : OrderedRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousAdd 𝕜
    A B C : Set E
    hB : IsExposed 𝕜 A B
    hC : IsExposed 𝕜 A C
    ⊢ IsExposed 𝕜 A (Inter.inter B C)
  -/
  rintro ⟨w, hwB, hwC⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : TopologicalSpace 𝕜
    inst✝⁴ : OrderedRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousAdd 𝕜
    A B C : Set E
    hB : IsExposed 𝕜 A B
    hC : IsExposed 𝕜 A C
    w : E
    hwB : Membership.mem B w
    hwC : Membership.mem C w
    ⊢ Exists fun l => Eq (Inter.inter B C) (setOf fun x => And (Membership.mem A x …
  -/
  obtain ⟨l₁, rfl⟩ := hB ⟨w, hwB⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : TopologicalSpace 𝕜
    inst✝⁴ : OrderedRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousAdd 𝕜
    A C : Set E
    hC : IsExposed 𝕜 A C
    w : E
    hwC : Membership.mem C w
    l₁ : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hB : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Member …
    hwB : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Memb …
    ⊢ Exists fun l => Eq (Inter.inter (setOf fun x => And (Membership.mem A x) (∀  …
  -/
  obtain ⟨l₂, rfl⟩ := hC ⟨w, hwC⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : TopologicalSpace 𝕜
    inst✝⁴ : OrderedRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousAdd 𝕜
    A : Set E
    w : E
    l₁ : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hB : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Member …
    hwB : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Memb …
    l₂ : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hC : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Member …
    hwC : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Memb …
    ⊢ Exists fun l => Eq (Inter.inter (setOf fun x => And (Membership.mem A x) (∀  …
  -/
  refine ⟨l₁ + l₂, Subset.antisymm ?_ ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : TopologicalSpace 𝕜
      inst✝⁴ : OrderedRing 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousAdd 𝕜
      A : Set E
      w : E
      l₁ : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
      hB : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Member …
      hwB : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Memb …
      l₂ : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
      hC : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Member …
      hwC : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Memb …
      ⊢ HasSubset.Subset (Inter.inter (setOf fun x => And (Membership.mem A x) (∀ (y …
    -/
  · rintro x ⟨⟨hxA, hxB⟩, ⟨-, hxC⟩⟩
    /-
      case intro.intro.intro.intro.refine_1.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : TopologicalSpace 𝕜
      inst✝⁴ : OrderedRing 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousAdd 𝕜
      A : Set E
      w : E
      l₁ : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
      hB : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Member …
      hwB : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Memb …
      l₂ : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
      hC : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Member …
      hwC : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Memb …
      x : E
      hxA : Membership.mem A x
      hxB : ∀ (y : E), Membership.mem A y → LE.le (l₁ y) (l₁ x)
      hxC : ∀ (y : E), Membership.mem A y → LE.le (l₂ y) (l₂ x)
      ⊢ Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Membersh …
    -/
    exact ⟨hxA, fun z hz => add_le_add (hxB z hz) (hxC z hz)⟩
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.intro.intro.refine_2
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : TopologicalSpace 𝕜
    inst✝⁴ : OrderedRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousAdd 𝕜
    A : Set E
    w : E
    l₁ : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hB : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Member …
    hwB : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Memb …
    l₂ : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hC : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Member …
    hwC : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Memb …
    ⊢ HasSubset.Subset (setOf fun x => And (Membership.mem A x) (∀ (y : E), Member …
  -/
  rintro x ⟨hxA, hx⟩
  /-
    case intro.intro.intro.intro.refine_2.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : TopologicalSpace 𝕜
    inst✝⁴ : OrderedRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousAdd 𝕜
    A : Set E
    w : E
    l₁ : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hB : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Member …
    hwB : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Memb …
    l₂ : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hC : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Member …
    hwC : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Memb …
    x : E
    hxA : Membership.mem A x
    hx : ∀ (y : E), Membership.mem A y → LE.le ((HAdd.hAdd l₁ l₂) y) ((HAdd.hAdd l …
    ⊢ Membership.mem (Inter.inter (setOf fun x => And (Membership.mem A x) (∀ (y : …
  -/
  refine ⟨⟨hxA, fun y hy => ?_⟩, hxA, fun y hy => ?_⟩
  · exact
      (add_le_add_iff_right (l₂ x)).1 ((add_le_add (hwB.2 y hy) (hwC.2 x hxA)).trans (hx w hwB.1))
  · exact
      (add_le_add_iff_left (l₁ x)).1 (le_trans (add_le_add (hwB.2 x hxA) (hwC.2 y hy)) (hx w hwB.1))


theorem sInter [ContinuousAdd 𝕜] {F : Finset (Set E)} (hF : F.Nonempty)
    (hAF : ∀ B ∈ F, IsExposed 𝕜 A B) : IsExposed 𝕜 A (⋂₀ F) := by
  classical
  induction F using Finset.induction with
  | empty => exfalso; exact Finset.not_nonempty_empty hF
  | @insert C F _ hF' =>
    rw [Finset.coe_insert, sInter_insert]
    obtain rfl | hFnemp := F.eq_empty_or_nonempty
    · rw [Finset.coe_empty, sInter_empty, inter_univ]
      exact hAF C (Finset.mem_singleton_self C)
    · exact (hAF C (Finset.mem_insert_self C F)).inter
        (hF' hFnemp fun B hB => hAF B (Finset.mem_insert_of_mem hB))


theorem inter_left (hC : IsExposed 𝕜 A C) (hCB : C ⊆ B) : IsExposed 𝕜 (A ∩ B) C := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : OrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A B C : Set E
    hC : IsExposed 𝕜 A C
    hCB : HasSubset.Subset C B
    ⊢ IsExposed 𝕜 (Inter.inter A B) C
  -/
  rintro ⟨w, hw⟩
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : OrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A B C : Set E
    hC : IsExposed 𝕜 A C
    hCB : HasSubset.Subset C B
    w : E
    hw : Membership.mem C w
    ⊢ Exists fun l => Eq C (setOf fun x => And (Membership.mem (Inter.inter A B) x …
  -/
  obtain ⟨l, rfl⟩ := hC ⟨w, hw⟩
  exact ⟨l, Subset.antisymm (fun x hx => ⟨⟨hx.1, hCB hx⟩, fun y hy => hx.2 y hy.1⟩)
    fun x ⟨⟨hxC, _⟩, hx⟩ => ⟨hxC, fun y hy => (hw.2 y hy).trans (hx w ⟨hC.subset hw, hCB hw⟩)⟩⟩


theorem inter_right (hC : IsExposed 𝕜 B C) (hCA : C ⊆ A) : IsExposed 𝕜 (A ∩ B) C := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : OrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A B C : Set E
    hC : IsExposed 𝕜 B C
    hCA : HasSubset.Subset C A
    ⊢ IsExposed 𝕜 (Inter.inter A B) C
  -/
  rw [inter_comm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : OrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A B C : Set E
    hC : IsExposed 𝕜 B C
    hCA : HasSubset.Subset C A
    ⊢ IsExposed 𝕜 (Inter.inter B A) C
  -/
  exact hC.inter_left hCA
  /-
    🎉 no goals
  -/


protected theorem isClosed [OrderClosedTopology 𝕜] {A B : Set E} (hAB : IsExposed 𝕜 A B)
    (hA : IsClosed A) : IsClosed B := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : TopologicalSpace 𝕜
    inst✝⁴ : OrderedRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : OrderClosedTopology 𝕜
    A B : Set E
    hAB : IsExposed 𝕜 A B
    hA : IsClosed A
    ⊢ IsClosed B
  -/
  obtain rfl | hB := B.eq_empty_or_nonempty
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : TopologicalSpace 𝕜
      inst✝⁴ : OrderedRing 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : OrderClosedTopology 𝕜
      A : Set E
      hA : IsClosed A
      hAB : IsExposed 𝕜 A EmptyCollection.emptyCollection
      ⊢ IsClosed EmptyCollection.emptyCollection
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : TopologicalSpace 𝕜
    inst✝⁴ : OrderedRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : OrderClosedTopology 𝕜
    A B : Set E
    hAB : IsExposed 𝕜 A B
    hA : IsClosed A
    hB : B.Nonempty
    ⊢ IsClosed B
  -/
  obtain ⟨l, a, rfl⟩ := hAB.eq_inter_halfSpace' hB
  /-
    case inr.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : TopologicalSpace 𝕜
    inst✝⁴ : OrderedRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : OrderClosedTopology 𝕜
    A : Set E
    hA : IsClosed A
    l : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    a : 𝕜
    hAB : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (LE.le a (l x)))
    hB : (setOf fun x => And (Membership.mem A x) (LE.le a (l x))).Nonempty
    ⊢ IsClosed (setOf fun x => And (Membership.mem A x) (LE.le a (l x)))
  -/
  exact hA.isClosed_le continuousOn_const l.continuous.continuousOn
  /-
    🎉 no goals
  -/


protected theorem isCompact [OrderClosedTopology 𝕜] [T2Space E] {A B : Set E}
    (hAB : IsExposed 𝕜 A B) (hA : IsCompact A) : IsCompact B :=
  hA.of_isClosed_subset (hAB.isClosed hA.isClosed) hAB.subset


/-- A point is exposed with respect to `A` iff there exists a hyperplane whose intersection with
`A` is exactly that point. -/
def Set.exposedPoints (A : Set E) : Set E :=
  { x ∈ A | ∃ l : E →L[𝕜] 𝕜, ∀ y ∈ A, l y ≤ l x ∧ (l x ≤ l y → y = x) }


theorem exposed_point_def :
    x ∈ A.exposedPoints 𝕜 ↔ x ∈ A ∧ ∃ l : E →L[𝕜] 𝕜, ∀ y ∈ A, l y ≤ l x ∧ (l x ≤ l y → y = x) :=
  Iff.rfl


theorem exposedPoints_subset : A.exposedPoints 𝕜 ⊆ A := fun _ hx => hx.1


@[simp]
theorem exposedPoints_empty : (∅ : Set E).exposedPoints 𝕜 = ∅ :=
  subset_empty_iff.1 exposedPoints_subset


/-- Exposed points exactly correspond to exposed singletons. -/
theorem mem_exposedPoints_iff_exposed_singleton : x ∈ A.exposedPoints 𝕜 ↔ IsExposed 𝕜 A {x} := by
  use fun ⟨hxA, l, hl⟩ _ =>
    ⟨l,
      Eq.symm <|
        eq_singleton_iff_unique_mem.2
          ⟨⟨hxA, fun y hy => (hl y hy).1⟩, fun z hz => (hl z hz.1).2 (hz.2 x hxA)⟩⟩
  /-
    case mpr
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : OrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A : Set E
    x : E
    ⊢ IsExposed 𝕜 A (Singleton.singleton x) → Membership.mem (Set.exposedPoints 𝕜  …
  -/
  rintro h
  /-
    case mpr
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : OrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A : Set E
    x : E
    h : IsExposed 𝕜 A (Singleton.singleton x)
    ⊢ Membership.mem (Set.exposedPoints 𝕜 A) x
  -/
  obtain ⟨l, hl⟩ := h ⟨x, mem_singleton _⟩
  /-
    case mpr.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : OrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A : Set E
    x : E
    h : IsExposed 𝕜 A (Singleton.singleton x)
    l : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hl : Eq (Singleton.singleton x) (setOf fun x => And (Membership.mem A x) (∀ (y …
    ⊢ Membership.mem (Set.exposedPoints 𝕜 A) x
  -/
  rw [eq_comm, eq_singleton_iff_unique_mem] at hl
  exact
    ⟨hl.1.1, l, fun y hy =>
      ⟨hl.1.2 y hy, fun hxy => hl.2 y ⟨hy, fun z hz => (hl.1.2 z hz).trans hxy⟩⟩⟩


protected theorem convex (hAB : IsExposed 𝕜 A B) (hA : Convex 𝕜 A) : Convex 𝕜 B := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : LinearOrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A B : Set E
    hAB : IsExposed 𝕜 A B
    hA : Convex 𝕜 A
    ⊢ Convex 𝕜 B
  -/
  obtain rfl | hB := B.eq_empty_or_nonempty
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : TopologicalSpace 𝕜
      inst✝³ : LinearOrderedRing 𝕜
      inst✝² : AddCommMonoid E
      inst✝¹ : TopologicalSpace E
      inst✝ : Module 𝕜 E
      A : Set E
      hA : Convex 𝕜 A
      hAB : IsExposed 𝕜 A EmptyCollection.emptyCollection
      ⊢ Convex 𝕜 EmptyCollection.emptyCollection
    -/
  · exact convex_empty
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : LinearOrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A B : Set E
    hAB : IsExposed 𝕜 A B
    hA : Convex 𝕜 A
    hB : B.Nonempty
    ⊢ Convex 𝕜 B
  -/
  obtain ⟨l, rfl⟩ := hAB hB
  exact fun x₁ hx₁ x₂ hx₂ a b ha hb hab =>
    ⟨hA hx₁.1 hx₂.1 ha hb hab, fun y hy =>
      ((l.toLinearMap.concaveOn convex_univ).convex_ge _ ⟨mem_univ _, hx₁.2 y hy⟩
          ⟨mem_univ _, hx₂.2 y hy⟩ ha hb hab).2⟩


protected theorem isExtreme (hAB : IsExposed 𝕜 A B) : IsExtreme 𝕜 A B := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : LinearOrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A B : Set E
    hAB : IsExposed 𝕜 A B
    ⊢ IsExtreme 𝕜 A B
  -/
  refine ⟨hAB.subset, fun x₁ hx₁A x₂ hx₂A x hxB hx => ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : LinearOrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A B : Set E
    hAB : IsExposed 𝕜 A B
    x₁ : E
    hx₁A : Membership.mem A x₁
    x₂ : E
    hx₂A : Membership.mem A x₂
    x : E
    hxB : Membership.mem B x
    hx : Membership.mem (openSegment 𝕜 x₁ x₂) x
    ⊢ And (Membership.mem B x₁) (Membership.mem B x₂)
  -/
  obtain ⟨l, rfl⟩ := hAB ⟨x, hxB⟩
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : LinearOrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A : Set E
    x₁ : E
    hx₁A : Membership.mem A x₁
    x₂ : E
    hx₂A : Membership.mem A x₂
    x : E
    hx : Membership.mem (openSegment 𝕜 x₁ x₂) x
    l : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hAB : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Membe …
    hxB : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Memb …
    ⊢ And (Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Mem …
  -/
  have hl : ConvexOn 𝕜 univ l := l.toLinearMap.convexOn convex_univ
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : LinearOrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A : Set E
    x₁ : E
    hx₁A : Membership.mem A x₁
    x₂ : E
    hx₂A : Membership.mem A x₂
    x : E
    hx : Membership.mem (openSegment 𝕜 x₁ x₂) x
    l : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hAB : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Membe …
    hxB : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Memb …
    hl : ConvexOn 𝕜 Set.univ ⇑l
    ⊢ And (Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Mem …
  -/
  have hlx₁ := hxB.2 x₁ hx₁A
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : LinearOrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A : Set E
    x₁ : E
    hx₁A : Membership.mem A x₁
    x₂ : E
    hx₂A : Membership.mem A x₂
    x : E
    hx : Membership.mem (openSegment 𝕜 x₁ x₂) x
    l : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hAB : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Membe …
    hxB : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Memb …
    hl : ConvexOn 𝕜 Set.univ ⇑l
    hlx₁ : LE.le (l x₁) (l x)
    ⊢ And (Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Mem …
  -/
  have hlx₂ := hxB.2 x₂ hx₂A
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : LinearOrderedRing 𝕜
    inst✝² : AddCommMonoid E
    inst✝¹ : TopologicalSpace E
    inst✝ : Module 𝕜 E
    A : Set E
    x₁ : E
    hx₁A : Membership.mem A x₁
    x₂ : E
    hx₂A : Membership.mem A x₂
    x : E
    hx : Membership.mem (openSegment 𝕜 x₁ x₂) x
    l : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hAB : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Membe …
    hxB : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Memb …
    hl : ConvexOn 𝕜 Set.univ ⇑l
    hlx₁ : LE.le (l x₁) (l x)
    hlx₂ : LE.le (l x₂) (l x)
    ⊢ And (Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Mem …
  -/
  refine ⟨⟨hx₁A, fun y hy => ?_⟩, ⟨hx₂A, fun y hy => ?_⟩⟩
    /-
      case intro.refine_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : TopologicalSpace 𝕜
      inst✝³ : LinearOrderedRing 𝕜
      inst✝² : AddCommMonoid E
      inst✝¹ : TopologicalSpace E
      inst✝ : Module 𝕜 E
      A : Set E
      x₁ : E
      hx₁A : Membership.mem A x₁
      x₂ : E
      hx₂A : Membership.mem A x₂
      x : E
      hx : Membership.mem (openSegment 𝕜 x₁ x₂) x
      l : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
      hAB : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Membe …
      hxB : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Memb …
      hl : ConvexOn 𝕜 Set.univ ⇑l
      hlx₁ : LE.le (l x₁) (l x)
      hlx₂ : LE.le (l x₂) (l x)
      y : E
      hy : Membership.mem A y
      ⊢ LE.le (l y) (l x₁)
    -/
  · rw [hlx₁.antisymm (hl.le_left_of_right_le (mem_univ _) (mem_univ _) hx hlx₂)]
    /-
      case intro.refine_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : TopologicalSpace 𝕜
      inst✝³ : LinearOrderedRing 𝕜
      inst✝² : AddCommMonoid E
      inst✝¹ : TopologicalSpace E
      inst✝ : Module 𝕜 E
      A : Set E
      x₁ : E
      hx₁A : Membership.mem A x₁
      x₂ : E
      hx₂A : Membership.mem A x₂
      x : E
      hx : Membership.mem (openSegment 𝕜 x₁ x₂) x
      l : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
      hAB : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Membe …
      hxB : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Memb …
      hl : ConvexOn 𝕜 Set.univ ⇑l
      hlx₁ : LE.le (l x₁) (l x)
      hlx₂ : LE.le (l x₂) (l x)
      y : E
      hy : Membership.mem A y
      ⊢ LE.le (l y) (l x)
    -/
    exact hxB.2 y hy
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : TopologicalSpace 𝕜
      inst✝³ : LinearOrderedRing 𝕜
      inst✝² : AddCommMonoid E
      inst✝¹ : TopologicalSpace E
      inst✝ : Module 𝕜 E
      A : Set E
      x₁ : E
      hx₁A : Membership.mem A x₁
      x₂ : E
      hx₂A : Membership.mem A x₂
      x : E
      hx : Membership.mem (openSegment 𝕜 x₁ x₂) x
      l : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
      hAB : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Membe …
      hxB : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Memb …
      hl : ConvexOn 𝕜 Set.univ ⇑l
      hlx₁ : LE.le (l x₁) (l x)
      hlx₂ : LE.le (l x₂) (l x)
      y : E
      hy : Membership.mem A y
      ⊢ LE.le (l y) (l x₂)
    -/
  · rw [hlx₂.antisymm (hl.le_right_of_left_le (mem_univ _) (mem_univ _) hx hlx₁)]
    /-
      case intro.refine_2
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : TopologicalSpace 𝕜
      inst✝³ : LinearOrderedRing 𝕜
      inst✝² : AddCommMonoid E
      inst✝¹ : TopologicalSpace E
      inst✝ : Module 𝕜 E
      A : Set E
      x₁ : E
      hx₁A : Membership.mem A x₁
      x₂ : E
      hx₂A : Membership.mem A x₂
      x : E
      hx : Membership.mem (openSegment 𝕜 x₁ x₂) x
      l : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
      hAB : IsExposed 𝕜 A (setOf fun x => And (Membership.mem A x) (∀ (y : E), Membe …
      hxB : Membership.mem (setOf fun x => And (Membership.mem A x) (∀ (y : E), Memb …
      hl : ConvexOn 𝕜 Set.univ ⇑l
      hlx₁ : LE.le (l x₁) (l x)
      hlx₂ : LE.le (l x₂) (l x)
      y : E
      hy : Membership.mem A y
      ⊢ LE.le (l y) (l x)
    -/
    exact hxB.2 y hy
    /-
      🎉 no goals
    -/


theorem exposedPoints_subset_extremePoints : A.exposedPoints 𝕜 ⊆ A.extremePoints 𝕜 := fun _ hx =>
  (mem_exposedPoints_iff_exposed_singleton.1 hx).isExtreme.mem_extremePoints


