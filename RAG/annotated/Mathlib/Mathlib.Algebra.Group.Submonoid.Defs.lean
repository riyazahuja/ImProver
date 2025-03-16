/-- `OneMemClass S M` says `S` is a type of subsets `s ≤ M`, such that `1 ∈ s` for all `s`. -/
class OneMemClass (S : Type*) (M : outParam Type*) [One M] [SetLike S M] : Prop where
  /-- By definition, if we have `OneMemClass S M`, we have `1 ∈ s` for all `s : S`. -/
  one_mem : ∀ s : S, (1 : M) ∈ s


/-- `ZeroMemClass S M` says `S` is a type of subsets `s ≤ M`, such that `0 ∈ s` for all `s`. -/
class ZeroMemClass (S : Type*) (M : outParam Type*) [Zero M] [SetLike S M] : Prop where
  /-- By definition, if we have `ZeroMemClass S M`, we have `0 ∈ s` for all `s : S`. -/
  zero_mem : ∀ s : S, (0 : M) ∈ s


/-- A submonoid of a monoid `M` is a subset containing 1 and closed under multiplication. -/
structure Submonoid (M : Type*) [MulOneClass M] extends Subsemigroup M where
  /-- A submonoid contains `1`. -/
  one_mem' : (1 : M) ∈ carrier


/-- `SubmonoidClass S M` says `S` is a type of subsets `s ≤ M` that contain `1`
and are closed under `(*)` -/
class SubmonoidClass (S : Type*) (M : outParam Type*) [MulOneClass M] [SetLike S M] extends
  MulMemClass S M, OneMemClass S M : Prop


/-- An additive submonoid of an additive monoid `M` is a subset containing 0 and
  closed under addition. -/
structure AddSubmonoid (M : Type*) [AddZeroClass M] extends AddSubsemigroup M where
  /-- An additive submonoid contains `0`. -/
  zero_mem' : (0 : M) ∈ carrier


/-- `AddSubmonoidClass S M` says `S` is a type of subsets `s ≤ M` that contain `0`
and are closed under `(+)` -/
class AddSubmonoidClass (S : Type*) (M : outParam Type*) [AddZeroClass M] [SetLike S M] extends
  AddMemClass S M, ZeroMemClass S M : Prop


@[to_additive (attr := aesop safe apply (rule_sets := [SetLike]))]
theorem pow_mem {M A} [Monoid M] [SetLike A M] [SubmonoidClass A M] {S : A} {x : M}
    (hx : x ∈ S) : ∀ n : ℕ, x ^ n ∈ S
  | 0 => by
    /-
      M : Type u_3
      A : Type u_4
      inst✝² : Monoid M
      inst✝¹ : SetLike A M
      inst✝ : SubmonoidClass A M
      S : A
      x : M
      hx : Membership.mem S x
      ⊢ Membership.mem S (HPow.hPow x 0)
    -/
    rw [pow_zero]
    /-
      M : Type u_3
      A : Type u_4
      inst✝² : Monoid M
      inst✝¹ : SetLike A M
      inst✝ : SubmonoidClass A M
      S : A
      x : M
      hx : Membership.mem S x
      ⊢ Membership.mem S 1
    -/
    exact OneMemClass.one_mem S
    /-
      🎉 no goals
    -/
  | n + 1 => by
    /-
      M : Type u_3
      A : Type u_4
      inst✝² : Monoid M
      inst✝¹ : SetLike A M
      inst✝ : SubmonoidClass A M
      S : A
      x : M
      hx : Membership.mem S x
      n : Nat
      ⊢ Membership.mem S (HPow.hPow x (HAdd.hAdd n 1))
    -/
    rw [pow_succ]
    /-
      M : Type u_3
      A : Type u_4
      inst✝² : Monoid M
      inst✝¹ : SetLike A M
      inst✝ : SubmonoidClass A M
      S : A
      x : M
      hx : Membership.mem S x
      n : Nat
      ⊢ Membership.mem S (HMul.hMul (HPow.hPow x n) x)
    -/
    exact mul_mem (pow_mem hx n) hx
    /-
      🎉 no goals
    -/


@[to_additive]
instance : SetLike (Submonoid M) M where
  coe s := s.carrier
                             /-
                               M : Type u_1
                               N : Type u_2
                               inst✝ : MulOneClass M
                               s : Set M
                               p q : Submonoid M
                               h : Eq ((fun s => s.carrier) p) ((fun s => s.carrier) q)
                               ⊢ Eq p q
                             -/
  coe_injective' p q h := by cases p; cases q; congr; exact SetLike.coe_injective' h
                                                      /-
                                                        🎉 no goals
                                                      -/


@[to_additive]
instance : SubmonoidClass (Submonoid M) M where
  one_mem := Submonoid.one_mem'
  mul_mem {s} := s.mul_mem'


@[to_additive (attr := simp)]
theorem mem_toSubsemigroup {s : Submonoid M} {x : M} : x ∈ s.toSubsemigroup ↔ x ∈ s :=
  Iff.rfl

-- Porting note: `x ∈ s.carrier` is now syntactically `x ∈ s.toSubsemigroup.carrier`,
-- which `simp` already simplifies to `x ∈ s.toSubsemigroup`. So we remove the `@[simp]` attribute
-- here, and instead add the simp lemma `mem_toSubsemigroup` to allow `simp` to do this exact
-- simplification transitively.

@[to_additive]
theorem mem_carrier {s : Submonoid M} {x : M} : x ∈ s.carrier ↔ x ∈ s :=
  Iff.rfl


@[to_additive (attr := simp)]
theorem mem_mk {s : Subsemigroup M} {x : M} (h_one) : x ∈ mk s h_one ↔ x ∈ s :=
  Iff.rfl


@[to_additive (attr := simp)]
theorem coe_set_mk {s : Subsemigroup M} (h_one) : (mk s h_one : Set M) = s :=
  rfl


@[to_additive (attr := simp)]
theorem mk_le_mk {s t : Subsemigroup M} (h_one) (h_one') : mk s h_one ≤ mk t h_one' ↔ s ≤ t :=
  Iff.rfl


/-- Two submonoids are equal if they have the same elements. -/
@[to_additive (attr := ext) "Two `AddSubmonoid`s are equal if they have the same elements."]
theorem ext {S T : Submonoid M} (h : ∀ x, x ∈ S ↔ x ∈ T) : S = T :=
  SetLike.ext h


/-- Copy a submonoid replacing `carrier` with a set that is equal to it. -/
@[to_additive "Copy an additive submonoid replacing `carrier` with a set that is equal to it."]
protected def copy (S : Submonoid M) (s : Set M) (hs : s = S) : Submonoid M where
  carrier := s
  one_mem' := show 1 ∈ s from hs.symm ▸ S.one_mem'
  mul_mem' := hs.symm ▸ S.mul_mem'


@[to_additive (attr := simp)]
theorem coe_copy {s : Set M} (hs : s = S) : (S.copy s hs : Set M) = s :=
  rfl


@[to_additive]
theorem copy_eq {s : Set M} (hs : s = S) : S.copy s hs = S :=
  SetLike.coe_injective hs


/-- A submonoid contains the monoid's 1. -/
@[to_additive "An `AddSubmonoid` contains the monoid's 0."]
protected theorem one_mem : (1 : M) ∈ S :=
  one_mem S


/-- A submonoid is closed under multiplication. -/
@[to_additive "An `AddSubmonoid` is closed under addition."]
protected theorem mul_mem {x y : M} : x ∈ S → y ∈ S → x * y ∈ S :=
  mul_mem


/-- The submonoid `M` of the monoid `M`. -/
@[to_additive "The additive submonoid `M` of the `AddMonoid M`."]
instance : Top (Submonoid M) :=
  ⟨{  carrier := Set.univ
      one_mem' := Set.mem_univ 1
      mul_mem' := fun _ _ => Set.mem_univ _ }⟩


/-- The trivial submonoid `{1}` of a monoid `M`. -/
@[to_additive "The trivial `AddSubmonoid` `{0}` of an `AddMonoid` `M`."]
instance : Bot (Submonoid M) :=
  ⟨{  carrier := {1}
      one_mem' := Set.mem_singleton 1
      mul_mem' := fun ha hb => by
        /-
          M : Type u_1
          N : Type u_2
          inst✝ : MulOneClass M
          s : Set M
          S : Submonoid M
          a✝ b✝ : M
          ha : Membership.mem (Singleton.singleton 1) a✝
          hb : Membership.mem (Singleton.singleton 1) b✝
          ⊢ Membership.mem (Singleton.singleton 1) (HMul.hMul a✝ b✝)
        -/
        simp only [Set.mem_singleton_iff] at *
        /-
          M : Type u_1
          N : Type u_2
          inst✝ : MulOneClass M
          s : Set M
          S : Submonoid M
          a✝ b✝ : M
          ha : Eq a✝ 1
          hb : Eq b✝ 1
          ⊢ Eq (HMul.hMul a✝ b✝) 1
        -/
        rw [ha, hb, mul_one] }⟩
        /-
          🎉 no goals
        -/


@[to_additive]
instance : Inhabited (Submonoid M) :=
  ⟨⊥⟩


@[to_additive (attr := simp)]
theorem mem_bot {x : M} : x ∈ (⊥ : Submonoid M) ↔ x = 1 :=
  Set.mem_singleton_iff


@[to_additive (attr := simp)]
theorem mem_top (x : M) : x ∈ (⊤ : Submonoid M) :=
  Set.mem_univ x


@[to_additive (attr := simp)]
theorem coe_top : ((⊤ : Submonoid M) : Set M) = Set.univ :=
  rfl


@[to_additive (attr := simp)]
theorem coe_bot : ((⊥ : Submonoid M) : Set M) = {1} :=
  rfl


/-- The inf of two submonoids is their intersection. -/
@[to_additive "The inf of two `AddSubmonoid`s is their intersection."]
instance : Min (Submonoid M) :=
  ⟨fun S₁ S₂ =>
    { carrier := S₁ ∩ S₂
      one_mem' := ⟨S₁.one_mem, S₂.one_mem⟩
      mul_mem' := fun ⟨hx, hx'⟩ ⟨hy, hy'⟩ => ⟨S₁.mul_mem hx hy, S₂.mul_mem hx' hy'⟩ }⟩


@[to_additive (attr := simp)]
theorem coe_inf (p p' : Submonoid M) : ((p ⊓ p' : Submonoid M) : Set M) = (p : Set M) ∩ p' :=
  rfl


@[to_additive (attr := simp)]
theorem mem_inf {p p' : Submonoid M} {x : M} : x ∈ p ⊓ p' ↔ x ∈ p ∧ x ∈ p' :=
  Iff.rfl


@[to_additive (attr := simp)]
theorem subsingleton_iff : Subsingleton (Submonoid M) ↔ Subsingleton M :=
  ⟨fun _ =>
    ⟨fun x y =>
      have : ∀ i : M, i = 1 := fun i =>
        mem_bot.mp <| Subsingleton.elim (⊤ : Submonoid M) ⊥ ▸ mem_top i
      (this x).trans (this y).symm⟩,
    fun _ =>
                                                                  /-
                                                                    M : Type u_1
                                                                    inst✝ : MulOneClass M
                                                                    x✝ : Subsingleton M
                                                                    x y : Submonoid M
                                                                    i : M
                                                                    ⊢ Iff (Membership.mem x 1) (Membership.mem y 1)
                                                                  -/
    ⟨fun x y => Submonoid.ext fun i => Subsingleton.elim 1 i ▸ by simp [Submonoid.one_mem]⟩⟩
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[to_additive (attr := simp)]
theorem nontrivial_iff : Nontrivial (Submonoid M) ↔ Nontrivial M :=
  not_iff_not.mp
    ((not_nontrivial_iff_subsingleton.trans subsingleton_iff).trans
      not_nontrivial_iff_subsingleton.symm)


@[to_additive]
instance [Subsingleton M] : Unique (Submonoid M) :=
  ⟨⟨⊥⟩, fun a => @Subsingleton.elim _ (subsingleton_iff.mpr ‹_›) a _⟩


@[to_additive]
instance [Nontrivial M] : Nontrivial (Submonoid M) :=
  nontrivial_iff.mpr ‹_›


/-- The submonoid of elements `x : M` such that `f x = g x` -/
@[to_additive "The additive submonoid of elements `x : M` such that `f x = g x`"]
def eqLocusM (f g : M →* N) : Submonoid M where
  carrier := { x | f x = g x }
                 /-
                   M : Type u_1
                   N : Type u_2
                   inst✝¹ : MulOneClass M
                   s : Set M
                   inst✝ : MulOneClass N
                   f g : MonoidHom M N
                   ⊢ Membership.mem { carrier := setOf fun x => Eq (f x) (g x), mul_mem' := ⋯ }.c …
                 -/
                                           /-
                                             M : Type u_1
                                             N : Type u_2
                                             inst✝¹ : MulOneClass M
                                             s : Set M
                                             inst✝ : MulOneClass N
                                             f g : MonoidHom M N
                                             a✝ b✝ : M
                                             hx : Eq (f a✝) (g a✝)
                                             hy : Eq (f b✝) (g b✝)
                                             ⊢ Membership.mem (setOf fun x => Eq (f x) (g x)) (HMul.hMul a✝ b✝)
                                           -/
  one_mem' := by rw [Set.mem_setOf_eq, f.map_one, g.map_one]
                                           /-
                                             🎉 no goals
                                           -/
                 /-
                   🎉 no goals
                 -/
  mul_mem' (hx : _ = _) (hy : _ = _) := by simp [*]


@[to_additive (attr := simp)]
theorem eqLocusM_same (f : M →* N) : f.eqLocusM f = ⊤ :=
  SetLike.ext fun _ => eq_self_iff_true _


@[to_additive]
theorem eq_of_eqOn_topM {f g : M →* N} (h : Set.EqOn f g (⊤ : Submonoid M)) : f = g :=
  ext fun _ => h trivial


/-- A submonoid of a monoid inherits a 1. -/
@[to_additive "An `AddSubmonoid` of an `AddMonoid` inherits a zero."]
instance one : One S' :=
  ⟨⟨1, OneMemClass.one_mem S'⟩⟩


@[to_additive (attr := simp, norm_cast)]
theorem coe_one : ((1 : S') : M₁) = 1 :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem coe_eq_one {x : S'} : (↑x : M₁) = 1 ↔ x = 1 :=
  (Subtype.ext_iff.symm : (x : M₁) = (1 : S') ↔ x = 1)


@[to_additive]
theorem one_def : (1 : S') = ⟨1, OneMemClass.one_mem S'⟩ :=
  rfl


/-- An `AddSubmonoid` of an `AddMonoid` inherits a scalar multiplication. -/
instance AddSubmonoidClass.nSMul {M} [AddMonoid M] {A : Type*} [SetLike A M]
    [AddSubmonoidClass A M] (S : A) : SMul ℕ S :=
  ⟨fun n a => ⟨n • a.1, nsmul_mem a.2 n⟩⟩


/-- A submonoid of a monoid inherits a power operator. -/
instance nPow {M} [Monoid M] {A : Type*} [SetLike A M] [SubmonoidClass A M] (S : A) : Pow S ℕ :=
  ⟨fun a n => ⟨a.1 ^ n, pow_mem a.2 n⟩⟩


attribute [to_additive existing nSMul] nPow


@[to_additive (attr := simp, norm_cast)]
theorem coe_pow {M} [Monoid M] {A : Type*} [SetLike A M] [SubmonoidClass A M] {S : A} (x : S)
    (n : ℕ) : ↑(x ^ n) = (x : M) ^ n :=
  rfl


@[to_additive (attr := simp)]
theorem mk_pow {M} [Monoid M] {A : Type*} [SetLike A M] [SubmonoidClass A M] {S : A} (x : M)
    (hx : x ∈ S) (n : ℕ) : (⟨x, hx⟩ : S) ^ n = ⟨x ^ n, pow_mem hx n⟩ :=
  rfl

-- Prefer subclasses of `Monoid` over subclasses of `SubmonoidClass`.

/-- A submonoid of a unital magma inherits a unital magma structure. -/
@[to_additive
      "An `AddSubmonoid` of a unital additive magma inherits a unital additive magma structure."]
instance (priority := 75) toMulOneClass {M : Type*} [MulOneClass M] {A : Type*} [SetLike A M]
    [SubmonoidClass A M] (S : A) : MulOneClass S :=
    Subtype.coe_injective.mulOneClass Subtype.val rfl (fun _ _ => rfl)

-- Prefer subclasses of `Monoid` over subclasses of `SubmonoidClass`.

/-- A submonoid of a monoid inherits a monoid structure. -/
@[to_additive "An `AddSubmonoid` of an `AddMonoid` inherits an `AddMonoid` structure."]
instance (priority := 75) toMonoid {M : Type*} [Monoid M] {A : Type*} [SetLike A M]
    [SubmonoidClass A M] (S : A) : Monoid S :=
  Subtype.coe_injective.monoid Subtype.val rfl (fun _ _ => rfl) (fun _ _ => rfl)

-- Prefer subclasses of `Monoid` over subclasses of `SubmonoidClass`.

/-- A submonoid of a `CommMonoid` is a `CommMonoid`. -/
@[to_additive "An `AddSubmonoid` of an `AddCommMonoid` is an `AddCommMonoid`."]
instance (priority := 75) toCommMonoid {M} [CommMonoid M] {A : Type*} [SetLike A M]
    [SubmonoidClass A M] (S : A) : CommMonoid S :=
  Subtype.coe_injective.commMonoid Subtype.val rfl (fun _ _ => rfl) fun _ _ => rfl


/-- The natural monoid hom from a submonoid of monoid `M` to `M`. -/
@[to_additive "The natural monoid hom from an `AddSubmonoid` of `AddMonoid` `M` to `M`."]
def subtype : S' →* M where
                                                            /-
                                                              M : Type u_1
                                                              N : Type u_2
                                                              A : Type u_3
                                                              inst✝¹ : MulOneClass M
                                                              inst✝ : SetLike A M
                                                              hA : SubmonoidClass A M
                                                              S' : A
                                                              x✝¹ x✝ : Subtype fun x => Membership.mem S' x
                                                              ⊢ Eq ({ toFun := Subtype.val, map_one' := ⋯ }.toFun (HMul.hMul x✝¹ x✝)) (HMul. …
                                                            -/
  toFun := Subtype.val; map_one' := rfl; map_mul' _ _ := by simp
                                                            /-
                                                              🎉 no goals
                                                            -/


@[to_additive (attr := simp)]
theorem coe_subtype : (SubmonoidClass.subtype S' : S' → M) = Subtype.val :=
  rfl


/-- A submonoid of a monoid inherits a multiplication. -/
@[to_additive "An `AddSubmonoid` of an `AddMonoid` inherits an addition."]
instance mul : Mul S :=
  ⟨fun a b => ⟨a.1 * b.1, S.mul_mem a.2 b.2⟩⟩


/-- A submonoid of a monoid inherits a 1. -/
@[to_additive "An `AddSubmonoid` of an `AddMonoid` inherits a zero."]
instance one : One S :=
  ⟨⟨_, S.one_mem⟩⟩


@[to_additive (attr := simp, norm_cast)]
theorem coe_mul (x y : S) : (↑(x * y) : M) = ↑x * ↑y :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem coe_one : ((1 : S) : M) = 1 :=
  rfl


@[to_additive (attr := simp)]
                                                               /-
                                                                 M : Type u_4
                                                                 inst✝ : MulOneClass M
                                                                 S : Submonoid M
                                                                 a : M
                                                                 ha : Membership.mem S a
                                                                 ⊢ Iff (Eq ⟨a, ha⟩ 1) (Eq a 1)
                                                               -/
lemma mk_eq_one {a : M} {ha} : (⟨a, ha⟩ : S) = 1 ↔ a = 1 := by simp [← SetLike.coe_eq_coe]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[to_additive (attr := simp)]
theorem mk_mul_mk (x y : M) (hx : x ∈ S) (hy : y ∈ S) :
    (⟨x, hx⟩ : S) * ⟨y, hy⟩ = ⟨x * y, S.mul_mem hx hy⟩ :=
  rfl


@[to_additive]
theorem mul_def (x y : S) : x * y = ⟨x * y, S.mul_mem x.2 y.2⟩ :=
  rfl


@[to_additive]
theorem one_def : (1 : S) = ⟨1, S.one_mem⟩ :=
  rfl


/-- A submonoid of a unital magma inherits a unital magma structure. -/
@[to_additive
      "An `AddSubmonoid` of a unital additive magma inherits a unital additive magma structure."]
instance toMulOneClass {M : Type*} [MulOneClass M] (S : Submonoid M) : MulOneClass S :=
  Subtype.coe_injective.mulOneClass Subtype.val rfl fun _ _ => rfl


@[to_additive]
protected theorem pow_mem {M : Type*} [Monoid M] (S : Submonoid M) {x : M} (hx : x ∈ S) (n : ℕ) :
    x ^ n ∈ S :=
  pow_mem hx n


/-- A submonoid of a monoid inherits a monoid structure. -/
@[to_additive "An `AddSubmonoid` of an `AddMonoid` inherits an `AddMonoid` structure."]
instance toMonoid {M : Type*} [Monoid M] (S : Submonoid M) : Monoid S :=
  Subtype.coe_injective.monoid Subtype.val rfl (fun _ _ => rfl) fun _ _ => rfl


/-- A submonoid of a `CommMonoid` is a `CommMonoid`. -/
@[to_additive "An `AddSubmonoid` of an `AddCommMonoid` is an `AddCommMonoid`."]
instance toCommMonoid {M} [CommMonoid M] (S : Submonoid M) : CommMonoid S :=
  Subtype.coe_injective.commMonoid Subtype.val rfl (fun _ _ => rfl) fun _ _ => rfl


/-- The natural monoid hom from a submonoid of monoid `M` to `M`. -/
@[to_additive "The natural monoid hom from an `AddSubmonoid` of `AddMonoid` `M` to `M`."]
def subtype : S →* M where
                                                            /-
                                                              M✝ : Type u_1
                                                              N : Type u_2
                                                              A : Type u_3
                                                              inst✝² : MulOneClass M✝
                                                              inst✝¹ : SetLike A M✝
                                                              hA : SubmonoidClass A M✝
                                                              S' : A
                                                              M : Type u_4
                                                              inst✝ : MulOneClass M
                                                              S : Submonoid M
                                                              x✝¹ x✝ : Subtype fun x => Membership.mem S x
                                                              ⊢ Eq ({ toFun := Subtype.val, map_one' := ⋯ }.toFun (HMul.hMul x✝¹ x✝)) (HMul. …
                                                            -/
  toFun := Subtype.val; map_one' := rfl; map_mul' _ _ := by simp
                                                            /-
                                                              🎉 no goals
                                                            -/


@[to_additive (attr := simp)]
theorem coe_subtype : ⇑S.subtype = Subtype.val :=
  rfl


