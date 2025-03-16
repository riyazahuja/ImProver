/-- Given a set of relations, `rels`, over a type `α`, `PresentedMonoid` constructs the monoid with
generators `x : α` and relations `rels` as a quotient of a congruence structure over rels. -/
@[to_additive "Given a set of relations, `rels`, over a type `α`, `PresentedAddMonoid` constructs
the monoid with generators `x : α` and relations `rels` as a quotient of an AddCon structure over
rels"]
def PresentedMonoid (rel : FreeMonoid α → FreeMonoid α → Prop) := (conGen rel).Quotient


@[to_additive]
instance {rels : FreeMonoid α → FreeMonoid α → Prop} : Monoid (PresentedMonoid rels) :=
  Con.monoid (conGen rels)


/-- The quotient map from the free monoid on `α` to the presented monoid with the same generators
and the given relations `rels`. -/
@[to_additive "The quotient map from the free additive monoid on `α` to the presented additive
monoid with the same generators and the given relations `rels`"]
def mk (rels : FreeMonoid α → FreeMonoid α → Prop) : FreeMonoid α →* PresentedMonoid rels where
  toFun := Quotient.mk (conGen rels).toSetoid
  map_one' := rfl
  map_mul' := fun _ _ => rfl


/-- `of` is the canonical map from `α` to a presented monoid with generators `x : α`. The term `x`
is mapped to the equivalence class of the image of `x` in `FreeMonoid α`. -/
@[to_additive "`of` is the canonical map from `α` to a presented additive monoid with generators
`x : α`. The term `x` is mapped to the equivalence class of the image of `x` in `FreeAddMonoid α`"]
def of (rels : FreeMonoid α → FreeMonoid α → Prop) (x : α) : PresentedMonoid rels :=
  mk rels (.of x)


local notation "P₁" => PresentedMonoid rels₁

local notation "P₂" => PresentedMonoid rels₂

local notation "P₃" => PresentedMonoid rels₃


@[to_additive (attr := elab_as_elim), induction_eliminator]
protected theorem inductionOn {δ : P₁ → Prop} (q : P₁) (h : ∀ a, δ (mk rels₁ a)) : δ q :=
  Quotient.ind h q


@[to_additive (attr := elab_as_elim)]
protected theorem inductionOn₂ {δ : P₁ → P₂ → Prop} (q₁ : P₁) (q₂ : P₂)
    (h : ∀ a b, δ (mk rels₁ a) (mk rels₂ b)) : δ q₁ q₂ :=
  Quotient.inductionOn₂ q₁ q₂ h


@[to_additive (attr := elab_as_elim)]
protected theorem inductionOn₃ {δ : P₁ → P₂ → P₃ → Prop} (q₁ : P₁)
    (q₂ : P₂) (q₃ : P₃) (h : ∀ a b c, δ (mk rels₁ a) (mk rels₂ b) (mk rels₃ c)) :
    δ q₁ q₂ q₃ :=
  Quotient.inductionOn₃ q₁ q₂ q₃ h


/-- The generators of a presented monoid generate the presented monoid. That is, the submonoid
closure of the set of generators equals `⊤`. -/
@[to_additive (attr := simp) "The generators of a presented additive monoid generate the presented
additive monoid. That is, the additive submonoid closure of the set of generators equals `⊤`"]
theorem closure_range_of (rels : FreeMonoid α → FreeMonoid α → Prop) :
    Submonoid.closure (Set.range (PresentedMonoid.of rels)) = ⊤ := by
  /-
    α : Type u_2
    rels : FreeMonoid α → FreeMonoid α → Prop
    ⊢ Eq (Submonoid.closure (Set.range (PresentedMonoid.of rels))) Top.top
  -/
  rw [Submonoid.eq_top_iff']
  /-
    α : Type u_2
    rels : FreeMonoid α → FreeMonoid α → Prop
    ⊢ ∀ (x : PresentedMonoid rels), Membership.mem (Submonoid.closure (Set.range ( …
  -/
  intro x
  /-
    α : Type u_2
    rels : FreeMonoid α → FreeMonoid α → Prop
    x : PresentedMonoid rels
    ⊢ Membership.mem (Submonoid.closure (Set.range (PresentedMonoid.of rels))) x
  -/
  induction' x with a
  /-
    case h
    α : Type u_2
    rels : FreeMonoid α → FreeMonoid α → Prop
    a : FreeMonoid α
    ⊢ Membership.mem (Submonoid.closure (Set.range (PresentedMonoid.of rels))) ((P …
  -/
  induction a
    /-
      case h.one
      α : Type u_2
      rels : FreeMonoid α → FreeMonoid α → Prop
      ⊢ Membership.mem (Submonoid.closure (Set.range (PresentedMonoid.of rels))) ((P …
    -/
  · exact Submonoid.one_mem _
    /-
      🎉 no goals
    -/
    /-
      case h.of
      α : Type u_2
      rels : FreeMonoid α → FreeMonoid α → Prop
      x✝ : α
      ⊢ Membership.mem (Submonoid.closure (Set.range (PresentedMonoid.of rels))) ((P …
    -/
  · rename_i x
    /-
      case h.of
      α : Type u_2
      rels : FreeMonoid α → FreeMonoid α → Prop
      x : α
      ⊢ Membership.mem (Submonoid.closure (Set.range (PresentedMonoid.of rels))) ((P …
    -/
    exact subset_closure (Exists.intro x rfl)
    /-
      🎉 no goals
    -/
  /-
    case h.mul
    α : Type u_2
    rels : FreeMonoid α → FreeMonoid α → Prop
    x✝ y✝ : FreeMonoid α
    a✝¹ : Membership.mem (Submonoid.closure (Set.range (PresentedMonoid.of rels))) …
    a✝ : Membership.mem (Submonoid.closure (Set.range (PresentedMonoid.of rels)))  …
    ⊢ Membership.mem (Submonoid.closure (Set.range (PresentedMonoid.of rels))) ((P …
  -/
  rename_i x y hx hy
  /-
    case h.mul
    α : Type u_2
    rels : FreeMonoid α → FreeMonoid α → Prop
    x y : FreeMonoid α
    hx : Membership.mem (Submonoid.closure (Set.range (PresentedMonoid.of rels)))  …
    hy : Membership.mem (Submonoid.closure (Set.range (PresentedMonoid.of rels)))  …
    ⊢ Membership.mem (Submonoid.closure (Set.range (PresentedMonoid.of rels))) ((P …
  -/
  exact Submonoid.mul_mem _ hx hy
  /-
    🎉 no goals
  -/


@[to_additive]
theorem surjective_mk {rels : FreeMonoid α → FreeMonoid α → Prop} :
    Function.Surjective (mk rels) := fun x ↦ PresentedMonoid.inductionOn x fun a ↦ .intro a rfl


/-- The extension of a map `f : α → M` that satisfies the given relations to a monoid homomorphism
from `PresentedMonoid rels → M`. -/
@[to_additive "The extension of a map `f : α → M` that satisfies the given relations to an
additive-monoid homomorphism from `PresentedAddMonoid rels → M`"]
def lift : PresentedMonoid rels →* M :=
  Con.lift _ (FreeMonoid.lift f) (Con.conGen_le h)


@[to_additive]
theorem toMonoid.unique (g : MonoidHom (conGen rels).Quotient M)
    (hg : ∀ a : α, g (of rels a) = f a) : g = lift f h :=
  Con.lift_unique (Con.conGen_le h) g (FreeMonoid.hom_eq fun x ↦ let_fun this := hg x; this)


@[to_additive (attr := simp)]
theorem lift_of {x : α} : lift f h (of rels x) = f x := rfl


@[to_additive (attr := ext)]
theorem ext {M : Type*} [Monoid M] (rels : FreeMonoid α → FreeMonoid α → Prop)
    {φ ψ : PresentedMonoid rels →* M} (hx : ∀ (x : α), φ (.of rels x) = ψ (.of rels x)) :
    φ = ψ := by
  /-
    α : Type u_2
    M : Type u_3
    inst✝ : Monoid M
    rels : FreeMonoid α → FreeMonoid α → Prop
    φ ψ : MonoidHom (PresentedMonoid rels) M
    hx : ∀ (x : α), Eq (φ (PresentedMonoid.of rels x)) (ψ (PresentedMonoid.of rels …
    ⊢ Eq φ ψ
  -/
  apply MonoidHom.eq_of_eqOn_denseM (closure_range_of _)
  /-
    α : Type u_2
    M : Type u_3
    inst✝ : Monoid M
    rels : FreeMonoid α → FreeMonoid α → Prop
    φ ψ : MonoidHom (PresentedMonoid rels) M
    hx : ∀ (x : α), Eq (φ (PresentedMonoid.of rels x)) (ψ (PresentedMonoid.of rels …
    ⊢ Set.EqOn (⇑φ) (⇑ψ) (Set.range (PresentedMonoid.of rels))
  -/
  apply eqOn_range.mpr
  /-
    α : Type u_2
    M : Type u_3
    inst✝ : Monoid M
    rels : FreeMonoid α → FreeMonoid α → Prop
    φ ψ : MonoidHom (PresentedMonoid rels) M
    hx : ∀ (x : α), Eq (φ (PresentedMonoid.of rels x)) (ψ (PresentedMonoid.of rels …
    ⊢ Eq (Function.comp (⇑φ) (PresentedMonoid.of rels)) (Function.comp (⇑ψ) (Prese …
  -/
  ext
  /-
    case h
    α : Type u_2
    M : Type u_3
    inst✝ : Monoid M
    rels : FreeMonoid α → FreeMonoid α → Prop
    φ ψ : MonoidHom (PresentedMonoid rels) M
    hx : ∀ (x : α), Eq (φ (PresentedMonoid.of rels x)) (ψ (PresentedMonoid.of rels …
    x✝ : α
    ⊢ Eq (Function.comp (⇑φ) (PresentedMonoid.of rels) x✝) (Function.comp (⇑ψ) (Pr …
  -/
  rw [Function.comp_apply]
  /-
    case h
    α : Type u_2
    M : Type u_3
    inst✝ : Monoid M
    rels : FreeMonoid α → FreeMonoid α → Prop
    φ ψ : MonoidHom (PresentedMonoid rels) M
    hx : ∀ (x : α), Eq (φ (PresentedMonoid.of rels x)) (ψ (PresentedMonoid.of rels …
    x✝ : α
    ⊢ Eq (φ (PresentedMonoid.of rels x✝)) (Function.comp (⇑ψ) (PresentedMonoid.of  …
  -/
  exact hx _
  /-
    🎉 no goals
  -/


