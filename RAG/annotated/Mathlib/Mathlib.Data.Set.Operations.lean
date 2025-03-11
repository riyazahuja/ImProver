compile_def% Union.union

compile_def% Inter.inter

compile_def% SDiff.sdiff

compile_def% HasCompl.compl

compile_def% EmptyCollection.emptyCollection

compile_def% Insert.insert

compile_def% Singleton.singleton


attribute [ext] Set.ext


@[simp, mfld_simps] theorem mem_setOf_eq {x : α} {p : α → Prop} : (x ∈ {y | p y}) = p x := rfl


@[simp, mfld_simps] theorem mem_univ (x : α) : x ∈ @univ α := trivial


instance : HasCompl (Set α) := ⟨fun s ↦ {x | x ∉ s}⟩


@[simp] theorem mem_compl_iff (s : Set α) (x : α) : x ∈ sᶜ ↔ x ∉ s := Iff.rfl


theorem diff_eq (s t : Set α) : s \ t = s ∩ tᶜ := rfl


@[simp] theorem mem_diff {s t : Set α} (x : α) : x ∈ s \ t ↔ x ∈ s ∧ x ∉ t := Iff.rfl


theorem mem_diff_of_mem {s t : Set α} {x : α} (h1 : x ∈ s) (h2 : x ∉ t) : x ∈ s \ t := ⟨h1, h2⟩

-- Porting note: I've introduced this abbreviation, with the `@[coe]` attribute,
-- so that `norm_cast` has something to index on.
-- It is currently an abbreviation so that instance coming from `Subtype` are available.
-- If you're interested in making it a `def`, as it probably should be,
-- you'll then need to create additional instances (and possibly prove lemmas about them).
-- The first error should appear below at `monotoneOn_iff_monotone`.

/-- Given the set `s`, `Elem s` is the `Type` of element of `s`. -/
@[coe, reducible] def Elem (s : Set α) : Type u := {x // x ∈ s}


/-- Coercion from a set to the corresponding subtype. -/
instance : CoeSort (Set α) (Type u) := ⟨Elem⟩


/-- The preimage of `s : Set β` by `f : α → β`, written `f ⁻¹' s`,
  is the set of `x : α` such that `f x ∈ s`. -/
def preimage (f : α → β) (s : Set β) : Set α := {x | f x ∈ s}


/-- `f ⁻¹' t` denotes the preimage of `t : Set β` under the function `f : α → β`. -/
infixl:80 " ⁻¹' " => preimage


@[simp, mfld_simps]
theorem mem_preimage {f : α → β} {s : Set β} {a : α} : a ∈ f ⁻¹' s ↔ f a ∈ s := Iff.rfl


/-- `f '' s` denotes the image of `s : Set α` under the function `f : α → β`. -/
infixl:80 " '' " => image


@[simp]
theorem mem_image (f : α → β) (s : Set α) (y : β) : y ∈ f '' s ↔ ∃ x ∈ s, f x = y :=
  Iff.rfl


@[mfld_simps]
theorem mem_image_of_mem (f : α → β) {x : α} {a : Set α} (h : x ∈ a) : f x ∈ f '' a :=
  ⟨_, h, rfl⟩


/-- Restriction of `f` to `s` factors through `s.imageFactorization f : s → f '' s`. -/
def imageFactorization (f : α → β) (s : Set α) : s → f '' s := fun p =>
  ⟨f p.1, mem_image_of_mem f p.2⟩


/-- `kernImage f s` is the set of `y` such that `f ⁻¹ y ⊆ s`. -/
def kernImage (f : α → β) (s : Set α) : Set β := {y | ∀ ⦃x⦄, f x = y → x ∈ s}


lemma subset_kernImage_iff {s : Set β} {t : Set α} {f : α → β} : s ⊆ kernImage f t ↔ f ⁻¹' s ⊆ t :=
  ⟨fun h _ hx ↦ h hx rfl,
    fun h _ hx y hy ↦ h (show f y ∈ s from hy.symm ▸ hx)⟩


/-- Range of a function.

This function is more flexible than `f '' univ`, as the image requires that the domain is in Type
and not an arbitrary Sort. -/
def range (f : ι → α) : Set α := {x | ∃ y, f y = x}


@[simp] theorem mem_range {x : α} : x ∈ range f ↔ ∃ y, f y = x := Iff.rfl


@[mfld_simps] theorem mem_range_self (i : ι) : f i ∈ range f := ⟨i, rfl⟩


/-- Any map `f : ι → α` factors through a map `rangeFactorization f : ι → range f`. -/
def rangeFactorization (f : ι → α) : ι → range f := fun i => ⟨f i, mem_range_self i⟩


/-- We can use the axiom of choice to pick a preimage for every element of `range f`. -/
noncomputable def rangeSplitting (f : α → β) : range f → α := fun x => x.2.choose

-- This can not be a `@[simp]` lemma because the head of the left hand side is a variable.

theorem apply_rangeSplitting (f : α → β) (x : range f) : f (rangeSplitting f x) = x :=
  x.2.choose_spec


@[simp]
theorem comp_rangeSplitting (f : α → β) : f ∘ rangeSplitting f = Subtype.val := by
  /-
    α : Type u
    β : Type v
    f : α → β
    ⊢ Eq (Function.comp f (Set.rangeSplitting f)) Subtype.val
  -/
  ext
  /-
    case h
    α : Type u
    β : Type v
    f : α → β
    x✝ : ↑(Set.range f)
    ⊢ Eq (Function.comp f (Set.rangeSplitting f) x✝) ↑x✝
  -/
  simp only [Function.comp_apply]
  /-
    case h
    α : Type u
    β : Type v
    f : α → β
    x✝ : ↑(Set.range f)
    ⊢ Eq (f (Set.rangeSplitting f x✝)) ↑x✝
  -/
  apply apply_rangeSplitting
  /-
    🎉 no goals
  -/


/-- The cartesian product `Set.prod s t` is the set of `(a, b)` such that `a ∈ s` and `b ∈ t`. -/
def prod (s : Set α) (t : Set β) : Set (α × β) := {p | p.1 ∈ s ∧ p.2 ∈ t}


@[default_instance]
instance instSProd : SProd (Set α) (Set β) (Set (α × β)) where
  sprod := Set.prod


theorem prod_eq (s : Set α) (t : Set β) : s ×ˢ t = Prod.fst ⁻¹' s ∩ Prod.snd ⁻¹' t := rfl


theorem mem_prod_eq : (p ∈ s ×ˢ t) = (p.1 ∈ s ∧ p.2 ∈ t) := rfl


@[simp, mfld_simps]
theorem mem_prod : p ∈ s ×ˢ t ↔ p.1 ∈ s ∧ p.2 ∈ t := .rfl


@[mfld_simps]
theorem prod_mk_mem_set_prod_eq : ((a, b) ∈ s ×ˢ t) = (a ∈ s ∧ b ∈ t) := rfl


theorem mk_mem_prod (ha : a ∈ s) (hb : b ∈ t) : (a, b) ∈ s ×ˢ t := ⟨ha, hb⟩


/-- `diagonal α` is the set of `α × α` consisting of all pairs of the form `(a, a)`. -/
def diagonal (α : Type*) : Set (α × α) := {p | p.1 = p.2}


theorem mem_diagonal (x : α) : (x, x) ∈ diagonal α := rfl


@[simp] theorem mem_diagonal_iff {x : α × α} : x ∈ diagonal α ↔ x.1 = x.2 := .rfl


/-- The off-diagonal of a set `s` is the set of pairs `(a, b)` with `a, b ∈ s` and `a ≠ b`. -/
def offDiag (s : Set α) : Set (α × α) := {x | x.1 ∈ s ∧ x.2 ∈ s ∧ x.1 ≠ x.2}


@[simp]
theorem mem_offDiag {x : α × α} {s : Set α} : x ∈ s.offDiag ↔ x.1 ∈ s ∧ x.2 ∈ s ∧ x.1 ≠ x.2 :=
  Iff.rfl


/-- Given an index set `ι` and a family of sets `t : Π i, Set (α i)`, `pi s t`
is the set of dependent functions `f : Πa, π a` such that `f i` belongs to `t i`
whenever `i ∈ s`. -/
def pi (s : Set ι) (t : ∀ i, Set (α i)) : Set (∀ i, α i) := {f | ∀ i ∈ s, f i ∈ t i}


@[simp] theorem mem_pi : f ∈ s.pi t ↔ ∀ i ∈ s, f i ∈ t i := .rfl


                                                           /-
                                                             ι : Type u_1
                                                             α : ι → Type u_2
                                                             t : (i : ι) → Set (α i)
                                                             f : (i : ι) → α i
                                                             ⊢ Iff (Membership.mem (Set.univ.pi t) f) (∀ (i : ι), Membership.mem (t i) (f i))
                                                           -/
theorem mem_univ_pi : f ∈ pi univ t ↔ ∀ i, f i ∈ t i := by simp
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- Two functions `f₁ f₂ : α → β` are equal on `s` if `f₁ x = f₂ x` for all `x ∈ s`. -/
def EqOn (f₁ f₂ : α → β) (s : Set α) : Prop := ∀ ⦃x⦄, x ∈ s → f₁ x = f₂ x


/-- `MapsTo f s t` means that the image of `s` is contained in `t`. -/
def MapsTo (f : α → β) (s : Set α) (t : Set β) : Prop := ∀ ⦃x⦄, x ∈ s → f x ∈ t


theorem mapsTo_image (f : α → β) (s : Set α) : MapsTo f s (f '' s) := fun _ ↦ mem_image_of_mem f


theorem mapsTo_preimage (f : α → β) (t : Set β) : MapsTo f (f ⁻¹' t) t := fun _ ↦ id


/-- Given a map `f` sending `s : Set α` into `t : Set β`, restrict domain of `f` to `s`
and the codomain to `t`. Same as `Subtype.map`. -/
def MapsTo.restrict (f : α → β) (s : Set α) (t : Set β) (h : MapsTo f s t) : s → t :=
  Subtype.map f h


/-- The restriction of a function onto the preimage of a set. -/
@[simps!]
def restrictPreimage (t : Set β) (f : α → β) : f ⁻¹' t → t :=
  (Set.mapsTo_preimage f t).restrict _ _ _


/-- `f` is injective on `s` if the restriction of `f` to `s` is injective. -/
def InjOn (f : α → β) (s : Set α) : Prop :=
  ∀ ⦃x₁ : α⦄, x₁ ∈ s → ∀ ⦃x₂ : α⦄, x₂ ∈ s → f x₁ = f x₂ → x₁ = x₂


/-- The graph of a function `f : α → β` on a set `s`. -/
def graphOn (f : α → β) (s : Set α) : Set (α × β) := (fun x ↦ (x, f x)) '' s


/-- `f` is surjective from `s` to `t` if `t` is contained in the image of `s`. -/
def SurjOn (f : α → β) (s : Set α) (t : Set β) : Prop := t ⊆ f '' s


/-- `f` is bijective from `s` to `t` if `f` is injective on `s` and `f '' s = t`. -/
def BijOn (f : α → β) (s : Set α) (t : Set β) : Prop := MapsTo f s t ∧ InjOn f s ∧ SurjOn f s t


/-- `g` is a left inverse to `f` on `s` means that `g (f x) = x` for all `x ∈ s`. -/
def LeftInvOn (f' : β → α) (f : α → β) (s : Set α) : Prop := ∀ ⦃x⦄, x ∈ s → f' (f x) = x


/-- `g` is a right inverse to `f` on `t` if `f (g x) = x` for all `x ∈ t`. -/
abbrev RightInvOn (f' : β → α) (f : α → β) (t : Set β) : Prop := LeftInvOn f f' t


/-- `g` is an inverse to `f` viewed as a map from `s` to `t` -/
def InvOn (g : β → α) (f : α → β) (s : Set α) (t : Set β) : Prop :=
  LeftInvOn g f s ∧ RightInvOn g f t


/-- The image of a binary function `f : α → β → γ` as a function `Set α → Set β → Set γ`.
Mathematically this should be thought of as the image of the corresponding function `α × β → γ`. -/
def image2 (f : α → β → γ) (s : Set α) (t : Set β) : Set γ := {c | ∃ a ∈ s, ∃ b ∈ t, f a b = c}


@[simp] theorem mem_image2 : c ∈ image2 f s t ↔ ∃ a ∈ s, ∃ b ∈ t, f a b = c := .rfl


theorem mem_image2_of_mem (ha : a ∈ s) (hb : b ∈ t) : f a b ∈ image2 f s t :=
  ⟨a, ha, b, hb, rfl⟩


/-- Given a set `s` of functions `α → β` and `t : Set α`, `seq s t` is the union of `f '' t` over
all `f ∈ s`. -/
def seq (s : Set (α → β)) (t : Set α) : Set β := image2 (fun f ↦ f) s t


@[simp]
theorem mem_seq_iff {s : Set (α → β)} {t : Set α} {b : β} :
    b ∈ seq s t ↔ ∃ f ∈ s, ∃ a ∈ t, (f : α → β) a = b :=
  Iff.rfl


lemma seq_eq_image2 (s : Set (α → β)) (t : Set α) : seq s t = image2 (fun f a ↦ f a) s t := rfl


