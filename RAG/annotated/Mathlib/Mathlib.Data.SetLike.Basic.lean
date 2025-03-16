/-- A class to indicate that there is a canonical injection between `A` and `Set B`.

This has the effect of giving terms of `A` elements of type `B` (through a `Membership`
instance) and a compatible coercion to `Type*` as a subtype.

Note: if `SetLike.coe` is a projection, implementers should create a simp lemma such as
```
@[simp] lemma mem_carrier {p : MySubobject X} : x ∈ p.carrier ↔ x ∈ (p : Set X) := Iff.rfl
```
to normalize terms.

If you declare an unbundled subclass of `SetLike`, for example:
```
class MulMemClass (S : Type*) (M : Type*) [Mul M] [SetLike S M] where
  ...
```
Then you should *not* repeat the `outParam` declaration so `SetLike` will supply the value instead.
This ensures your subclass will not have issues with synthesis of the `[Mul M]` parameter starting
before the value of `M` is known.
-/
@[notation_class * carrier Simps.findCoercionArgs]
class SetLike (A : Type*) (B : outParam Type*) where
  /-- The coercion from a term of a `SetLike` to its corresponding `Set`. -/
  protected coe : A → Set B
  /-- The coercion from a term of a `SetLike` to its corresponding `Set` is injective. -/
  protected coe_injective' : Function.Injective coe


instance : CoeTC A (Set B) where coe := SetLike.coe


instance (priority := 100) instMembership : Membership B A :=
  ⟨fun p x => x ∈ (p : Set B)⟩


instance (priority := 100) : CoeSort A (Type _) :=
  ⟨fun p => { x : B // x ∈ p }⟩


/-- For terms that match the `CoeSort` instance's body, pretty print as `↥S`
rather than as `{ x // x ∈ S }`. The discriminating feature is that membership
uses the `SetLike.instMembership` instance. -/
@[app_delab Subtype]
def delabSubtypeSetLike : Delab := whenPPOption getPPNotation do
  let #[_, .lam n _ body _] := (← getExpr).getAppArgs | failure
  guard <| body.isAppOf ``Membership.mem
  let #[_, _, inst, _, .bvar 0] := body.getAppArgs | failure
  guard <| inst.isAppOfArity ``instMembership 3
  let S ← withAppArg <| withBindingBody n <| withNaryArg 3 delab
  `(↥$S)


@[simp, norm_cast]
theorem coe_sort_coe : ((p : Set B) : Type _) = p :=
  rfl


protected theorem «exists» {q : p → Prop} : (∃ x, q x) ↔ ∃ (x : B) (h : x ∈ p), q ⟨x, ‹_›⟩ :=
  SetCoe.exists


protected theorem «forall» {q : p → Prop} : (∀ x, q x) ↔ ∀ (x : B) (h : x ∈ p), q ⟨x, ‹_›⟩ :=
  SetCoe.forall


theorem coe_injective : Function.Injective (SetLike.coe : A → Set B) := fun _ _ h =>
  SetLike.coe_injective' h


@[simp, norm_cast]
theorem coe_set_eq : (p : Set B) = q ↔ p = q :=
  coe_injective.eq_iff


@[norm_cast] lemma coe_ne_coe : (p : Set B) ≠ q ↔ p ≠ q := coe_injective.ne_iff


theorem ext' (h : (p : Set B) = q) : p = q :=
  coe_injective h


theorem ext'_iff : p = q ↔ (p : Set B) = q :=
  coe_set_eq.symm


/-- Note: implementers of `SetLike` must copy this lemma in order to tag it with `@[ext]`. -/
theorem ext (h : ∀ x, x ∈ p ↔ x ∈ q) : p = q :=
  coe_injective <| Set.ext h


theorem ext_iff : p = q ↔ ∀ x, x ∈ p ↔ x ∈ q :=
  coe_injective.eq_iff.symm.trans Set.ext_iff


@[simp]
theorem mem_coe {x : B} : x ∈ (p : Set B) ↔ x ∈ p :=
  Iff.rfl


@[simp, norm_cast]
theorem coe_eq_coe {x y : p} : (x : B) = y ↔ x = y :=
  Subtype.ext_iff_val.symm

-- Porting note: this is not necessary anymore due to the way coercions work


@[simp]
theorem coe_mem (x : p) : (x : B) ∈ p :=
  x.2


@[aesop 5% apply (rule_sets := [SetLike])]
lemma mem_of_subset {s : Set B} (hp : s ⊆ p) {x : B} (hx : x ∈ s) : x ∈ p := hp hx

-- Porting note: removed `@[simp]` because `simpNF` linter complained

protected theorem eta (x : p) (hx : (x : B) ∈ p) : (⟨x, hx⟩ : p) = x := rfl


@[simp] lemma setOf_mem_eq (a : A) : {b | b ∈ a} = a := rfl


instance (priority := 100) instPartialOrder : PartialOrder A :=
  { PartialOrder.lift (SetLike.coe : A → Set B) coe_injective with
    le := fun H K => ∀ ⦃x⦄, x ∈ H → x ∈ K }


theorem le_def {S T : A} : S ≤ T ↔ ∀ ⦃x : B⦄, x ∈ S → x ∈ T :=
  Iff.rfl


@[simp, norm_cast] lemma coe_subset_coe {S T : A} : (S : Set B) ⊆ T ↔ S ≤ T := .rfl

@[simp, norm_cast] lemma coe_ssubset_coe {S T : A} : (S : Set B) ⊂ T ↔ S < T := .rfl


@[gcongr] protected alias ⟨_, GCongr.coe_subset_coe⟩ := coe_subset_coe

@[gcongr] protected alias ⟨_, GCongr.coe_ssubset_coe⟩ := coe_ssubset_coe


@[mono]
theorem coe_mono : Monotone (SetLike.coe : A → Set B) := fun _ _ => coe_subset_coe.mpr


@[mono]
theorem coe_strictMono : StrictMono (SetLike.coe : A → Set B) := fun _ _ => coe_ssubset_coe.mpr


theorem not_le_iff_exists : ¬p ≤ q ↔ ∃ x ∈ p, x ∉ q :=
  Set.not_subset


theorem exists_of_lt : p < q → ∃ x ∈ q, x ∉ p :=
  Set.exists_of_ssubset


theorem lt_iff_le_and_exists : p < q ↔ p ≤ q ∧ ∃ x ∈ q, x ∉ p := by
  /-
    A : Type u_1
    B : Type u_2
    i : SetLike A B
    p q : A
    ⊢ Iff (LT.lt p q) (And (LE.le p q) (Exists fun x => And (Membership.mem q x) ( …
  -/
  rw [lt_iff_le_not_le, not_le_iff_exists]
  /-
    🎉 no goals
  -/


