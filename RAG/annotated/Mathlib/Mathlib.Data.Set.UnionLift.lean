/-- Given a union of sets `iUnion S`, define a function on the Union by defining
it on each component, and proving that it agrees on the intersections. -/
@[nolint unusedArguments]
noncomputable def iUnionLift (S : ι → Set α) (f : ∀ i, S i → β)
    (_ : ∀ (i j) (x : α) (hxi : x ∈ S i) (hxj : x ∈ S j), f i ⟨x, hxi⟩ = f j ⟨x, hxj⟩) (T : Set α)
    (hT : T ⊆ iUnion S) (x : T) : β :=
  let i := Classical.indefiniteDescription _ (mem_iUnion.1 (hT x.prop))
  f i ⟨x, i.prop⟩


@[simp]
theorem iUnionLift_mk {i : ι} (x : S i) (hx : (x : α) ∈ T) :
    iUnionLift S f hf T hT ⟨x, hx⟩ = f i x := hf _ i x _ _


theorem iUnionLift_inclusion {i : ι} (x : S i) (h : S i ⊆ T) :
    iUnionLift S f hf T hT (Set.inclusion h x) = f i x :=
  iUnionLift_mk x _


theorem iUnionLift_of_mem (x : T) {i : ι} (hx : (x : α) ∈ S i) :
                                                 /-
                                                   α : Type u_1
                                                   ι : Sort u_3
                                                   β : Sort u_2
                                                   S : ι → Set α
                                                   f : (i : ι) → ↑(S i) → β
                                                   hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
                                                   T : Set α
                                                   hT : HasSubset.Subset T (Set.iUnion S)
                                                   x : ↑T
                                                   i : ι
                                                   hx : Membership.mem (S i) ↑x
                                                   ⊢ Eq (Set.iUnionLift S f hf T hT x) (f i ⟨↑x, hx⟩)
                                                 -/
    iUnionLift S f hf T hT x = f i ⟨x, hx⟩ := by cases' x with x hx; exact hf _ _ _ _ _
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem preimage_iUnionLift (t : Set β) :
    iUnionLift S f hf T hT ⁻¹' t =
      inclusion hT ⁻¹' (⋃ i, inclusion (subset_iUnion S i) '' (f i ⁻¹' t)) := by
  /-
    α : Type u_1
    ι : Sort u_3
    β : Type u_2
    S : ι → Set α
    f : (i : ι) → ↑(S i) → β
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    T : Set α
    hT : HasSubset.Subset T (Set.iUnion S)
    t : Set β
    ⊢ Eq (Set.preimage (Set.iUnionLift S f hf T hT) t) (Set.preimage (Set.inclusio …
  -/
  ext x
  /-
    case h
    α : Type u_1
    ι : Sort u_3
    β : Type u_2
    S : ι → Set α
    f : (i : ι) → ↑(S i) → β
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    T : Set α
    hT : HasSubset.Subset T (Set.iUnion S)
    t : Set β
    x : ↑T
    ⊢ Iff (Membership.mem (Set.preimage (Set.iUnionLift S f hf T hT) t) x) (Member …
  -/
  simp only [mem_preimage, mem_iUnion, mem_image]
  /-
    case h
    α : Type u_1
    ι : Sort u_3
    β : Type u_2
    S : ι → Set α
    f : (i : ι) → ↑(S i) → β
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    T : Set α
    hT : HasSubset.Subset T (Set.iUnion S)
    t : Set β
    x : ↑T
    ⊢ Iff (Membership.mem t (Set.iUnionLift S f hf T hT x)) (Exists fun i => Exist …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      ι : Sort u_3
      β : Type u_2
      S : ι → Set α
      f : (i : ι) → ↑(S i) → β
      hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
      T : Set α
      hT : HasSubset.Subset T (Set.iUnion S)
      t : Set β
      x : ↑T
      ⊢ Membership.mem t (Set.iUnionLift S f hf T hT x) → Exists fun i => Exists fun …
    -/
  · rcases mem_iUnion.1 (hT x.prop) with ⟨i, hi⟩
    /-
      case h.mp.intro
      α : Type u_1
      ι : Sort u_3
      β : Type u_2
      S : ι → Set α
      f : (i : ι) → ↑(S i) → β
      hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
      T : Set α
      hT : HasSubset.Subset T (Set.iUnion S)
      t : Set β
      x : ↑T
      i : ι
      hi : Membership.mem (S i) ↑x
      ⊢ Membership.mem t (Set.iUnionLift S f hf T hT x) → Exists fun i => Exists fun …
    -/
    refine fun h => ⟨i, ⟨x, hi⟩, ?_, rfl⟩
    /-
      case h.mp.intro
      α : Type u_1
      ι : Sort u_3
      β : Type u_2
      S : ι → Set α
      f : (i : ι) → ↑(S i) → β
      hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
      T : Set α
      hT : HasSubset.Subset T (Set.iUnion S)
      t : Set β
      x : ↑T
      i : ι
      hi : Membership.mem (S i) ↑x
      h : Membership.mem t (Set.iUnionLift S f hf T hT x)
      ⊢ Membership.mem t (f i ⟨↑x, hi⟩)
    -/
    rwa [iUnionLift_of_mem x hi] at h
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      ι : Sort u_3
      β : Type u_2
      S : ι → Set α
      f : (i : ι) → ↑(S i) → β
      hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
      T : Set α
      hT : HasSubset.Subset T (Set.iUnion S)
      t : Set β
      x : ↑T
      ⊢ (Exists fun i => Exists fun x_1 => And (Membership.mem t (f i x_1)) (Eq (Set …
    -/
  · rintro ⟨i, ⟨y, hi⟩, h, hxy⟩
    /-
      case h.mpr.intro.intro.mk.intro
      α : Type u_1
      ι : Sort u_3
      β : Type u_2
      S : ι → Set α
      f : (i : ι) → ↑(S i) → β
      hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
      T : Set α
      hT : HasSubset.Subset T (Set.iUnion S)
      t : Set β
      x : ↑T
      i : ι
      y : α
      hi : Membership.mem (S i) y
      h : Membership.mem t (f i ⟨y, hi⟩)
      hxy : Eq (Set.inclusion ⋯ ⟨y, hi⟩) (Set.inclusion hT x)
      ⊢ Membership.mem t (Set.iUnionLift S f hf T hT x)
    -/
    obtain rfl : y = x := congr_arg Subtype.val hxy
    /-
      case h.mpr.intro.intro.mk.intro
      α : Type u_1
      ι : Sort u_3
      β : Type u_2
      S : ι → Set α
      f : (i : ι) → ↑(S i) → β
      hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
      T : Set α
      hT : HasSubset.Subset T (Set.iUnion S)
      t : Set β
      x : ↑T
      i : ι
      hi : Membership.mem (S i) ↑x
      h : Membership.mem t (f i ⟨↑x, hi⟩)
      hxy : Eq (Set.inclusion ⋯ ⟨↑x, hi⟩) (Set.inclusion hT x)
      ⊢ Membership.mem t (Set.iUnionLift S f hf T hT x)
    -/
    rwa [iUnionLift_of_mem x hi]
    /-
      🎉 no goals
    -/


/-- `iUnionLift_const` is useful for proving that `iUnionLift` is a homomorphism
  of algebraic structures when defined on the Union of algebraic subobjects.
  For example, it could be used to prove that the lift of a collection
  of group homomorphisms on a union of subgroups preserves `1`. -/
theorem iUnionLift_const (c : T) (ci : ∀ i, S i) (hci : ∀ i, (ci i : α) = c) (cβ : β)
    (h : ∀ i, f i (ci i) = cβ) : iUnionLift S f hf T hT c = cβ := by
  /-
    α : Type u_1
    ι : Sort u_3
    β : Sort u_2
    S : ι → Set α
    f : (i : ι) → ↑(S i) → β
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    T : Set α
    hT : HasSubset.Subset T (Set.iUnion S)
    c : ↑T
    ci : (i : ι) → ↑(S i)
    hci : ∀ (i : ι), Eq ↑(ci i) ↑c
    cβ : β
    h : ∀ (i : ι), Eq (f i (ci i)) cβ
    ⊢ Eq (Set.iUnionLift S f hf T hT c) cβ
  -/
  let ⟨i, hi⟩ := Set.mem_iUnion.1 (hT c.prop)
  /-
    α : Type u_1
    ι : Sort u_3
    β : Sort u_2
    S : ι → Set α
    f : (i : ι) → ↑(S i) → β
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    T : Set α
    hT : HasSubset.Subset T (Set.iUnion S)
    c : ↑T
    ci : (i : ι) → ↑(S i)
    hci : ∀ (i : ι), Eq ↑(ci i) ↑c
    cβ : β
    h : ∀ (i : ι), Eq (f i (ci i)) cβ
    i : ι
    hi : Membership.mem (S i) ↑c
    ⊢ Eq (Set.iUnionLift S f hf T hT c) cβ
  -/
  have : ci i = ⟨c, hi⟩ := Subtype.ext (hci i)
  /-
    α : Type u_1
    ι : Sort u_3
    β : Sort u_2
    S : ι → Set α
    f : (i : ι) → ↑(S i) → β
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    T : Set α
    hT : HasSubset.Subset T (Set.iUnion S)
    c : ↑T
    ci : (i : ι) → ↑(S i)
    hci : ∀ (i : ι), Eq ↑(ci i) ↑c
    cβ : β
    h : ∀ (i : ι), Eq (f i (ci i)) cβ
    i : ι
    hi : Membership.mem (S i) ↑c
    this : Eq (ci i) ⟨↑c, hi⟩
    ⊢ Eq (Set.iUnionLift S f hf T hT c) cβ
  -/
  rw [iUnionLift_of_mem _ hi, ← this, h]
  /-
    🎉 no goals
  -/


/-- `iUnionLift_unary` is useful for proving that `iUnionLift` is a homomorphism
  of algebraic structures when defined on the Union of algebraic subobjects.
  For example, it could be used to prove that the lift of a collection
  of linear_maps on a union of submodules preserves scalar multiplication. -/
theorem iUnionLift_unary (u : T → T) (ui : ∀ i, S i → S i)
    (hui :
      ∀ (i) (x : S i),
        u (Set.inclusion (show S i ⊆ T from hT'.symm ▸ Set.subset_iUnion S i) x) =
          Set.inclusion (show S i ⊆ T from hT'.symm ▸ Set.subset_iUnion S i) (ui i x))
    (uβ : β → β) (h : ∀ (i) (x : S i), f i (ui i x) = uβ (f i x)) (x : T) :
    iUnionLift S f hf T (le_of_eq hT') (u x) = uβ (iUnionLift S f hf T (le_of_eq hT') x) := by
  /-
    α : Type u_1
    ι : Sort u_2
    β : Sort u_3
    S : ι → Set α
    f : (i : ι) → ↑(S i) → β
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    T : Set α
    hT' : Eq T (Set.iUnion S)
    u : ↑T → ↑T
    ui : (i : ι) → ↑(S i) → ↑(S i)
    hui : ∀ (i : ι) (x : ↑(S i)), Eq (u (Set.inclusion ⋯ x)) (Set.inclusion ⋯ (ui  …
    uβ : β → β
    h : ∀ (i : ι) (x : ↑(S i)), Eq (f i (ui i x)) (uβ (f i x))
    x : ↑T
    ⊢ Eq (Set.iUnionLift S f hf T ⋯ (u x)) (uβ (Set.iUnionLift S f hf T ⋯ x))
  -/
  subst hT'
  /-
    α : Type u_1
    ι : Sort u_2
    β : Sort u_3
    S : ι → Set α
    f : (i : ι) → ↑(S i) → β
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    ui : (i : ι) → ↑(S i) → ↑(S i)
    uβ : β → β
    h : ∀ (i : ι) (x : ↑(S i)), Eq (f i (ui i x)) (uβ (f i x))
    u : ↑(Set.iUnion S) → ↑(Set.iUnion S)
    hui : ∀ (i : ι) (x : ↑(S i)), Eq (u (Set.inclusion ⋯ x)) (Set.inclusion ⋯ (ui  …
    x : ↑(Set.iUnion S)
    ⊢ Eq (Set.iUnionLift S f hf (Set.iUnion S) ⋯ (u x)) (uβ (Set.iUnionLift S f hf …
  -/
  cases' Set.mem_iUnion.1 x.prop with i hi
  /-
    case intro
    α : Type u_1
    ι : Sort u_2
    β : Sort u_3
    S : ι → Set α
    f : (i : ι) → ↑(S i) → β
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    ui : (i : ι) → ↑(S i) → ↑(S i)
    uβ : β → β
    h : ∀ (i : ι) (x : ↑(S i)), Eq (f i (ui i x)) (uβ (f i x))
    u : ↑(Set.iUnion S) → ↑(Set.iUnion S)
    hui : ∀ (i : ι) (x : ↑(S i)), Eq (u (Set.inclusion ⋯ x)) (Set.inclusion ⋯ (ui  …
    x : ↑(Set.iUnion S)
    i : ι
    hi : Membership.mem (S i) ↑x
    ⊢ Eq (Set.iUnionLift S f hf (Set.iUnion S) ⋯ (u x)) (uβ (Set.iUnionLift S f hf …
  -/
  rw [iUnionLift_of_mem x hi, ← h i]
  have : x = Set.inclusion (Set.subset_iUnion S i) ⟨x, hi⟩ := by
    cases x
    rfl
  /-
    case intro
    α : Type u_1
    ι : Sort u_2
    β : Sort u_3
    S : ι → Set α
    f : (i : ι) → ↑(S i) → β
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    ui : (i : ι) → ↑(S i) → ↑(S i)
    uβ : β → β
    h : ∀ (i : ι) (x : ↑(S i)), Eq (f i (ui i x)) (uβ (f i x))
    u : ↑(Set.iUnion S) → ↑(Set.iUnion S)
    hui : ∀ (i : ι) (x : ↑(S i)), Eq (u (Set.inclusion ⋯ x)) (Set.inclusion ⋯ (ui  …
    x : ↑(Set.iUnion S)
    i : ι
    hi : Membership.mem (S i) ↑x
    this : Eq x (Set.inclusion ⋯ ⟨↑x, hi⟩)
    ⊢ Eq (Set.iUnionLift S f hf (Set.iUnion S) ⋯ (u x)) (f i (ui i ⟨↑x, hi⟩))
  -/
  conv_lhs => rw [this, hui, iUnionLift_inclusion]
  /-
    🎉 no goals
  -/


/-- `iUnionLift_binary` is useful for proving that `iUnionLift` is a homomorphism
  of algebraic structures when defined on the Union of algebraic subobjects.
  For example, it could be used to prove that the lift of a collection
  of group homomorphisms on a union of subgroups preserves `*`. -/
theorem iUnionLift_binary (dir : Directed (· ≤ ·) S) (op : T → T → T) (opi : ∀ i, S i → S i → S i)
    (hopi :
      ∀ i x y,
        Set.inclusion (show S i ⊆ T from hT'.symm ▸ Set.subset_iUnion S i) (opi i x y) =
          op (Set.inclusion (show S i ⊆ T from hT'.symm ▸ Set.subset_iUnion S i) x)
            (Set.inclusion (show S i ⊆ T from hT'.symm ▸ Set.subset_iUnion S i) y))
    (opβ : β → β → β) (h : ∀ (i) (x y : S i), f i (opi i x y) = opβ (f i x) (f i y)) (x y : T) :
    iUnionLift S f hf T (le_of_eq hT') (op x y) =
      opβ (iUnionLift S f hf T (le_of_eq hT') x) (iUnionLift S f hf T (le_of_eq hT') y) := by
  /-
    α : Type u_1
    ι : Sort u_2
    β : Sort u_3
    S : ι → Set α
    f : (i : ι) → ↑(S i) → β
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    T : Set α
    hT' : Eq T (Set.iUnion S)
    dir : Directed (fun x1 x2 => LE.le x1 x2) S
    op : ↑T → ↑T → ↑T
    opi : (i : ι) → ↑(S i) → ↑(S i) → ↑(S i)
    hopi : ∀ (i : ι) (x y : ↑(S i)), Eq (Set.inclusion ⋯ (opi i x y)) (op (Set.inc …
    opβ : β → β → β
    h : ∀ (i : ι) (x y : ↑(S i)), Eq (f i (opi i x y)) (opβ (f i x) (f i y))
    x y : ↑T
    ⊢ Eq (Set.iUnionLift S f hf T ⋯ (op x y)) (opβ (Set.iUnionLift S f hf T ⋯ x) ( …
  -/
  subst hT'
  /-
    α : Type u_1
    ι : Sort u_2
    β : Sort u_3
    S : ι → Set α
    f : (i : ι) → ↑(S i) → β
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    dir : Directed (fun x1 x2 => LE.le x1 x2) S
    opi : (i : ι) → ↑(S i) → ↑(S i) → ↑(S i)
    opβ : β → β → β
    h : ∀ (i : ι) (x y : ↑(S i)), Eq (f i (opi i x y)) (opβ (f i x) (f i y))
    op : ↑(Set.iUnion S) → ↑(Set.iUnion S) → ↑(Set.iUnion S)
    hopi : ∀ (i : ι) (x y : ↑(S i)), Eq (Set.inclusion ⋯ (opi i x y)) (op (Set.inc …
    x y : ↑(Set.iUnion S)
    ⊢ Eq (Set.iUnionLift S f hf (Set.iUnion S) ⋯ (op x y)) (opβ (Set.iUnionLift S  …
  -/
  cases' Set.mem_iUnion.1 x.prop with i hi
  /-
    case intro
    α : Type u_1
    ι : Sort u_2
    β : Sort u_3
    S : ι → Set α
    f : (i : ι) → ↑(S i) → β
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    dir : Directed (fun x1 x2 => LE.le x1 x2) S
    opi : (i : ι) → ↑(S i) → ↑(S i) → ↑(S i)
    opβ : β → β → β
    h : ∀ (i : ι) (x y : ↑(S i)), Eq (f i (opi i x y)) (opβ (f i x) (f i y))
    op : ↑(Set.iUnion S) → ↑(Set.iUnion S) → ↑(Set.iUnion S)
    hopi : ∀ (i : ι) (x y : ↑(S i)), Eq (Set.inclusion ⋯ (opi i x y)) (op (Set.inc …
    x y : ↑(Set.iUnion S)
    i : ι
    hi : Membership.mem (S i) ↑x
    ⊢ Eq (Set.iUnionLift S f hf (Set.iUnion S) ⋯ (op x y)) (opβ (Set.iUnionLift S  …
  -/
  cases' Set.mem_iUnion.1 y.prop with j hj
  /-
    case intro.intro
    α : Type u_1
    ι : Sort u_2
    β : Sort u_3
    S : ι → Set α
    f : (i : ι) → ↑(S i) → β
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    dir : Directed (fun x1 x2 => LE.le x1 x2) S
    opi : (i : ι) → ↑(S i) → ↑(S i) → ↑(S i)
    opβ : β → β → β
    h : ∀ (i : ι) (x y : ↑(S i)), Eq (f i (opi i x y)) (opβ (f i x) (f i y))
    op : ↑(Set.iUnion S) → ↑(Set.iUnion S) → ↑(Set.iUnion S)
    hopi : ∀ (i : ι) (x y : ↑(S i)), Eq (Set.inclusion ⋯ (opi i x y)) (op (Set.inc …
    x y : ↑(Set.iUnion S)
    i : ι
    hi : Membership.mem (S i) ↑x
    j : ι
    hj : Membership.mem (S j) ↑y
    ⊢ Eq (Set.iUnionLift S f hf (Set.iUnion S) ⋯ (op x y)) (opβ (Set.iUnionLift S  …
  -/
  rcases dir i j with ⟨k, hik, hjk⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    ι : Sort u_2
    β : Sort u_3
    S : ι → Set α
    f : (i : ι) → ↑(S i) → β
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    dir : Directed (fun x1 x2 => LE.le x1 x2) S
    opi : (i : ι) → ↑(S i) → ↑(S i) → ↑(S i)
    opβ : β → β → β
    h : ∀ (i : ι) (x y : ↑(S i)), Eq (f i (opi i x y)) (opβ (f i x) (f i y))
    op : ↑(Set.iUnion S) → ↑(Set.iUnion S) → ↑(Set.iUnion S)
    hopi : ∀ (i : ι) (x y : ↑(S i)), Eq (Set.inclusion ⋯ (opi i x y)) (op (Set.inc …
    x y : ↑(Set.iUnion S)
    i : ι
    hi : Membership.mem (S i) ↑x
    j : ι
    hj : Membership.mem (S j) ↑y
    k : ι
    hik : LE.le (S i) (S k)
    hjk : LE.le (S j) (S k)
    ⊢ Eq (Set.iUnionLift S f hf (Set.iUnion S) ⋯ (op x y)) (opβ (Set.iUnionLift S  …
  -/
  rw [iUnionLift_of_mem x (hik hi), iUnionLift_of_mem y (hjk hj), ← h k]
  have hx : x = Set.inclusion (Set.subset_iUnion S k) ⟨x, hik hi⟩ := by
    cases x
    rfl
  have hy : y = Set.inclusion (Set.subset_iUnion S k) ⟨y, hjk hj⟩ := by
    cases y
    rfl
  have hxy : (Set.inclusion (Set.subset_iUnion S k) (opi k ⟨x, hik hi⟩ ⟨y, hjk hj⟩) : α) ∈ S k :=
    (opi k ⟨x, hik hi⟩ ⟨y, hjk hj⟩).prop
  /-
    case intro.intro.intro.intro
    α : Type u_1
    ι : Sort u_2
    β : Sort u_3
    S : ι → Set α
    f : (i : ι) → ↑(S i) → β
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    dir : Directed (fun x1 x2 => LE.le x1 x2) S
    opi : (i : ι) → ↑(S i) → ↑(S i) → ↑(S i)
    opβ : β → β → β
    h : ∀ (i : ι) (x y : ↑(S i)), Eq (f i (opi i x y)) (opβ (f i x) (f i y))
    op : ↑(Set.iUnion S) → ↑(Set.iUnion S) → ↑(Set.iUnion S)
    hopi : ∀ (i : ι) (x y : ↑(S i)), Eq (Set.inclusion ⋯ (opi i x y)) (op (Set.inc …
    x y : ↑(Set.iUnion S)
    i : ι
    hi : Membership.mem (S i) ↑x
    j : ι
    hj : Membership.mem (S j) ↑y
    k : ι
    hik : LE.le (S i) (S k)
    hjk : LE.le (S j) (S k)
    hx : Eq x (Set.inclusion ⋯ ⟨↑x, ⋯⟩)
    hy : Eq y (Set.inclusion ⋯ ⟨↑y, ⋯⟩)
    hxy : Membership.mem (S k) ↑(Set.inclusion ⋯ (opi k ⟨↑x, ⋯⟩ ⟨↑y, ⋯⟩))
    ⊢ Eq (Set.iUnionLift S f hf (Set.iUnion S) ⋯ (op x y)) (f k (opi k ⟨↑x, ⋯⟩ ⟨↑y …
  -/
  conv_lhs => rw [hx, hy, ← hopi, iUnionLift_of_mem _ hxy]
  /-
    🎉 no goals
  -/


/-- Glue together functions defined on each of a collection `S` of sets that cover a type. See
  also `Set.iUnionLift`.   -/
noncomputable def liftCover (S : ι → Set α) (f : ∀ i, S i → β)
    (hf : ∀ (i j) (x : α) (hxi : x ∈ S i) (hxj : x ∈ S j), f i ⟨x, hxi⟩ = f j ⟨x, hxj⟩)
    (hS : iUnion S = univ) (a : α) : β :=
  iUnionLift S f hf univ hS.symm.subset ⟨a, trivial⟩


@[simp]
theorem liftCover_coe {i : ι} (x : S i) : liftCover S f hf hS x = f i x :=
  iUnionLift_mk x _


theorem liftCover_of_mem {i : ι} {x : α} (hx : (x : α) ∈ S i) :
    liftCover S f hf hS x = f i ⟨x, hx⟩ :=
  iUnionLift_of_mem (⟨x, trivial⟩ : {_z // True}) hx


theorem preimage_liftCover (t : Set β) : liftCover S f hf hS ⁻¹' t = ⋃ i, (↑) '' (f i ⁻¹' t) := by
  /-
    α : Type u_1
    ι : Sort u_3
    β : Type u_2
    S : ι → Set α
    f : (i : ι) → ↑(S i) → β
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    hS : Eq (Set.iUnion S) Set.univ
    t : Set β
    ⊢ Eq (Set.preimage (Set.liftCover S f hf hS) t) (Set.iUnion fun i => Set.image …
  -/
  change (iUnionLift S f hf univ hS.symm.subset ∘ fun a => ⟨a, mem_univ a⟩) ⁻¹' t = _
  /-
    α : Type u_1
    ι : Sort u_3
    β : Type u_2
    S : ι → Set α
    f : (i : ι) → ↑(S i) → β
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    hS : Eq (Set.iUnion S) Set.univ
    t : Set β
    ⊢ Eq (Set.preimage (Function.comp (Set.iUnionLift S f hf Set.univ ⋯) fun a =>  …
  -/
  rw [preimage_comp, preimage_iUnionLift]
  /-
    α : Type u_1
    ι : Sort u_3
    β : Type u_2
    S : ι → Set α
    f : (i : ι) → ↑(S i) → β
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (S i) x) (hxj : Membership.mem  …
    hS : Eq (Set.iUnion S) Set.univ
    t : Set β
    ⊢ Eq (Set.preimage (fun a => ⟨a, ⋯⟩) (Set.preimage (Set.inclusion ⋯) (Set.iUni …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


