/-- `MulMemClass S M` says `S` is a type of sets `s : Set M` that are closed under `(*)` -/
class MulMemClass (S : Type*) (M : outParam Type*) [Mul M] [SetLike S M] : Prop where
  /-- A substructure satisfying `MulMemClass` is closed under multiplication. -/
  mul_mem : ∀ {s : S} {a b : M}, a ∈ s → b ∈ s → a * b ∈ s


/-- `AddMemClass S M` says `S` is a type of sets `s : Set M` that are closed under `(+)` -/
class AddMemClass (S : Type*) (M : outParam Type*) [Add M] [SetLike S M] : Prop where
  /-- A substructure satisfying `AddMemClass` is closed under addition. -/
  add_mem : ∀ {s : S} {a b : M}, a ∈ s → b ∈ s → a + b ∈ s


/-- A subsemigroup of a magma `M` is a subset closed under multiplication. -/
structure Subsemigroup (M : Type*) [Mul M] where
  /-- The carrier of a subsemigroup. -/
  carrier : Set M
  /-- The product of two elements of a subsemigroup belongs to the subsemigroup. -/
  mul_mem' {a b} : a ∈ carrier → b ∈ carrier → a * b ∈ carrier


/-- An additive subsemigroup of an additive magma `M` is a subset closed under addition. -/
structure AddSubsemigroup (M : Type*) [Add M] where
  /-- The carrier of an additive subsemigroup. -/
  carrier : Set M
  /-- The sum of two elements of an additive subsemigroup belongs to the subsemigroup. -/
  add_mem' {a b} : a ∈ carrier → b ∈ carrier → a + b ∈ carrier


@[to_additive]
instance : SetLike (Subsemigroup M) M :=
                                         /-
                                           M : Type u_1
                                           N : Type u_2
                                           inst✝ : Mul M
                                           s : Set M
                                           p q : Subsemigroup M
                                           h : Eq p.carrier q.carrier
                                           ⊢ Eq p q
                                         -/
  ⟨Subsemigroup.carrier, fun p q h => by cases p; cases q; congr⟩
                                                           /-
                                                             🎉 no goals
                                                           -/


@[to_additive]
instance : MulMemClass (Subsemigroup M) M where mul_mem := fun {_ _ _} => Subsemigroup.mul_mem' _


@[to_additive (attr := simp)]
theorem mem_carrier {s : Subsemigroup M} {x : M} : x ∈ s.carrier ↔ x ∈ s :=
  Iff.rfl


@[to_additive (attr := simp)]
theorem mem_mk {s : Set M} {x : M} (h_mul) : x ∈ mk s h_mul ↔ x ∈ s :=
  Iff.rfl


@[to_additive (attr := simp, norm_cast)]
theorem coe_set_mk (s : Set M) (h_mul) : (mk s h_mul : Set M) = s :=
  rfl


@[to_additive (attr := simp)]
theorem mk_le_mk {s t : Set M} (h_mul) (h_mul') : mk s h_mul ≤ mk t h_mul' ↔ s ⊆ t :=
  Iff.rfl


/-- Two subsemigroups are equal if they have the same elements. -/
@[to_additive (attr := ext) "Two `AddSubsemigroup`s are equal if they have the same elements."]
theorem ext {S T : Subsemigroup M} (h : ∀ x, x ∈ S ↔ x ∈ T) : S = T :=
  SetLike.ext h


/-- Copy a subsemigroup replacing `carrier` with a set that is equal to it. -/
@[to_additive "Copy an additive subsemigroup replacing `carrier` with a set that is equal to it."]
protected def copy (S : Subsemigroup M) (s : Set M) (hs : s = S) :
    Subsemigroup M where
  carrier := s
  mul_mem' := hs.symm ▸ S.mul_mem'


@[to_additive (attr := simp)]
theorem coe_copy {s : Set M} (hs : s = S) : (S.copy s hs : Set M) = s :=
  rfl


@[to_additive]
theorem copy_eq {s : Set M} (hs : s = S) : S.copy s hs = S :=
  SetLike.coe_injective hs


/-- A subsemigroup is closed under multiplication. -/
@[to_additive "An `AddSubsemigroup` is closed under addition."]
protected theorem mul_mem {x y : M} : x ∈ S → y ∈ S → x * y ∈ S :=
  Subsemigroup.mul_mem' S


/-- The subsemigroup `M` of the magma `M`. -/
@[to_additive "The additive subsemigroup `M` of the magma `M`."]
instance : Top (Subsemigroup M) :=
  ⟨{  carrier := Set.univ
      mul_mem' := fun _ _ => Set.mem_univ _ }⟩


/-- The trivial subsemigroup `∅` of a magma `M`. -/
@[to_additive "The trivial `AddSubsemigroup` `∅` of an additive magma `M`."]
instance : Bot (Subsemigroup M) :=
  ⟨{  carrier := ∅
      mul_mem' := False.elim }⟩


@[to_additive]
instance : Inhabited (Subsemigroup M) :=
  ⟨⊥⟩


@[to_additive]
theorem not_mem_bot {x : M} : x ∉ (⊥ : Subsemigroup M) :=
  Set.not_mem_empty x


@[to_additive (attr := simp)]
theorem mem_top (x : M) : x ∈ (⊤ : Subsemigroup M) :=
  Set.mem_univ x


@[to_additive (attr := simp)]
theorem coe_top : ((⊤ : Subsemigroup M) : Set M) = Set.univ :=
  rfl


@[to_additive (attr := simp)]
theorem coe_bot : ((⊥ : Subsemigroup M) : Set M) = ∅ :=
  rfl


/-- The inf of two subsemigroups is their intersection. -/
@[to_additive "The inf of two `AddSubsemigroup`s is their intersection."]
instance : Min (Subsemigroup M) :=
  ⟨fun S₁ S₂ =>
    { carrier := S₁ ∩ S₂
      mul_mem' := fun ⟨hx, hx'⟩ ⟨hy, hy'⟩ => ⟨S₁.mul_mem hx hy, S₂.mul_mem hx' hy'⟩ }⟩


@[to_additive (attr := simp)]
theorem coe_inf (p p' : Subsemigroup M) : ((p ⊓ p' : Subsemigroup M) : Set M) = (p : Set M) ∩ p' :=
  rfl


@[to_additive (attr := simp)]
theorem mem_inf {p p' : Subsemigroup M} {x : M} : x ∈ p ⊓ p' ↔ x ∈ p ∧ x ∈ p' :=
  Iff.rfl


@[to_additive]
theorem subsingleton_of_subsingleton [Subsingleton (Subsemigroup M)] : Subsingleton M := by
  /-
    M : Type u_1
    inst✝¹ : Mul M
    inst✝ : Subsingleton (Subsemigroup M)
    ⊢ Subsingleton M
  -/
  constructor; intro x y
  /-
    case allEq
    M : Type u_1
    inst✝¹ : Mul M
    inst✝ : Subsingleton (Subsemigroup M)
    x y : M
    ⊢ Eq x y
  -/
  have : ∀ a : M, a ∈ (⊥ : Subsemigroup M) := by simp [Subsingleton.elim (⊥ : Subsemigroup M) ⊤]
  /-
    case allEq
    M : Type u_1
    inst✝¹ : Mul M
    inst✝ : Subsingleton (Subsemigroup M)
    x y : M
    this : ∀ (a : M), Membership.mem Bot.bot a
    ⊢ Eq x y
  -/
  exact absurd (this x) not_mem_bot
  /-
    🎉 no goals
  -/


@[to_additive]
instance [hn : Nonempty M] : Nontrivial (Subsemigroup M) :=
  ⟨⟨⊥, ⊤, fun h => by
      /-
        M : Type u_1
        N : Type u_2
        inst✝ : Mul M
        s : Set M
        S : Subsemigroup M
        hn : Nonempty M
        h : Eq Bot.bot Top.top
        ⊢ False
      -/
      obtain ⟨x⟩ := id hn
      /-
        case intro
        M : Type u_1
        N : Type u_2
        inst✝ : Mul M
        s : Set M
        S : Subsemigroup M
        hn : Nonempty M
        h : Eq Bot.bot Top.top
        x : M
        ⊢ False
      -/
      refine absurd (?_ : x ∈ ⊥) not_mem_bot
      /-
        case intro
        M : Type u_1
        N : Type u_2
        inst✝ : Mul M
        s : Set M
        S : Subsemigroup M
        hn : Nonempty M
        h : Eq Bot.bot Top.top
        x : M
        ⊢ Membership.mem Bot.bot x
      -/
      simp [h]⟩⟩
      /-
        🎉 no goals
      -/


/-- The subsemigroup of elements `x : M` such that `f x = g x` -/
@[to_additive "The additive subsemigroup of elements `x : M` such that `f x = g x`"]
def eqLocus (f g : M →ₙ* N) : Subsemigroup M where
  carrier := { x | f x = g x }
                                           /-
                                             M : Type u_1
                                             N : Type u_2
                                             inst✝¹ : Mul M
                                             s : Set M
                                             inst✝ : Mul N
                                             f g : MulHom M N
                                             a✝ b✝ : M
                                             hx : Eq (f a✝) (g a✝)
                                             hy : Eq (f b✝) (g b✝)
                                             ⊢ Membership.mem (setOf fun x => Eq (f x) (g x)) (HMul.hMul a✝ b✝)
                                           -/
  mul_mem' (hx : _ = _) (hy : _ = _) := by simp [*]
                                           /-
                                             🎉 no goals
                                           -/


@[to_additive]
theorem eq_of_eqOn_top {f g : M →ₙ* N} (h : Set.EqOn f g (⊤ : Subsemigroup M)) : f = g :=
  ext fun _ => h trivial


/-- A submagma of a magma inherits a multiplication. -/
@[to_additive "An additive submagma of an additive magma inherits an addition."]
instance (priority := 900) mul : Mul S' :=
  ⟨fun a b => ⟨a.1 * b.1, mul_mem a.2 b.2⟩⟩

-- lower priority so later simp lemmas are used first; to appease simp_nf

@[to_additive (attr := simp low, norm_cast)]
theorem coe_mul (x y : S') : (↑(x * y) : M) = ↑x * ↑y :=
  rfl

-- lower priority so later simp lemmas are used first; to appease simp_nf

@[to_additive (attr := simp low)]
theorem mk_mul_mk (x y : M) (hx : x ∈ S') (hy : y ∈ S') :
    (⟨x, hx⟩ : S') * ⟨y, hy⟩ = ⟨x * y, mul_mem hx hy⟩ :=
  rfl


@[to_additive]
theorem mul_def (x y : S') : x * y = ⟨x * y, mul_mem x.2 y.2⟩ :=
  rfl


/-- A subsemigroup of a semigroup inherits a semigroup structure. -/
@[to_additive "An `AddSubsemigroup` of an `AddSemigroup` inherits an `AddSemigroup` structure."]
instance toSemigroup {M : Type*} [Semigroup M] {A : Type*} [SetLike A M] [MulMemClass A M]
    (S : A) : Semigroup S :=
  Subtype.coe_injective.semigroup Subtype.val fun _ _ => rfl


/-- A subsemigroup of a `CommSemigroup` is a `CommSemigroup`. -/
@[to_additive "An `AddSubsemigroup` of an `AddCommSemigroup` is an `AddCommSemigroup`."]
instance toCommSemigroup {M} [CommSemigroup M] {A : Type*} [SetLike A M] [MulMemClass A M]
    (S : A) : CommSemigroup S :=
  Subtype.coe_injective.commSemigroup Subtype.val fun _ _ => rfl


/-- The natural semigroup hom from a subsemigroup of semigroup `M` to `M`. -/
@[to_additive "The natural semigroup hom from an `AddSubsemigroup` of
`AddSubsemigroup` `M` to `M`."]
def subtype : S' →ₙ* M where
  toFun := Subtype.val; map_mul' := fun _ _ => rfl


@[to_additive (attr := simp)]
theorem coe_subtype : (MulMemClass.subtype S' : S' → M) = Subtype.val :=
  rfl


