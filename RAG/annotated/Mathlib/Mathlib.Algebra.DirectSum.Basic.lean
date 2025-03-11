/-- `DirectSum ι β` is the direct sum of a family of additive commutative monoids `β i`.

Note: `open DirectSum` will enable the notation `⨁ i, β i` for `DirectSum ι β`. -/
def DirectSum [∀ i, AddCommMonoid (β i)] : Type _ :=
  -- Porting note: Failed to synthesize
  -- Π₀ i, β i deriving AddCommMonoid, Inhabited
  -- See https://github.com/leanprover-community/mathlib4/issues/5020
  Π₀ i, β i

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): Added inhabited instance manually

instance [∀ i, AddCommMonoid (β i)] : Inhabited (DirectSum ι β) :=
  inferInstanceAs (Inhabited (Π₀ i, β i))

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): Added addCommMonoid instance manually

instance [∀ i, AddCommMonoid (β i)] : AddCommMonoid (DirectSum ι β) :=
  inferInstanceAs (AddCommMonoid (Π₀ i, β i))


instance [∀ i, AddCommMonoid (β i)] : DFunLike (DirectSum ι β) _ fun i : ι => β i :=
  inferInstanceAs (DFunLike (Π₀ i, β i) _ _)


instance [∀ i, AddCommMonoid (β i)] : CoeFun (DirectSum ι β) fun _ => ∀ i : ι, β i :=
  inferInstanceAs (CoeFun (Π₀ i, β i) fun _ => ∀ i : ι, β i)


/-- `⨁ i, f i` is notation for `DirectSum _ f` and equals the direct sum of `fun i ↦ f i`.
Taking the direct sum over multiple arguments is possible, e.g. `⨁ (i) (j), f i j`. -/
scoped[DirectSum] notation3 "⨁ "(...)", "r:(scoped f => DirectSum _ f) => r

-- Porting note: The below recreates some of the lean3 notation, not fully yet
-- section
-- open Batteries.ExtendedBinder
-- syntax (name := bigdirectsum) "⨁ " extBinders ", " term : term
-- macro_rules (kind := bigdirectsum)
--   | `(⨁ $_:ident, $y:ident → $z:ident) => `(DirectSum _ (fun $y ↦ $z))
--   | `(⨁ $x:ident, $p) => `(DirectSum _ (fun $x ↦ $p))
--   | `(⨁ $_:ident : $t:ident, $p) => `(DirectSum _ (fun $t ↦ $p))
--   | `(⨁ ($x:ident) ($y:ident), $p) => `(DirectSum _ (fun $x ↦ fun $y ↦ $p))
-- end


instance [DecidableEq ι] [∀ i, AddCommMonoid (β i)] [∀ i, DecidableEq (β i)] :
    DecidableEq (DirectSum ι β) :=
  inferInstanceAs <| DecidableEq (Π₀ i, β i)


instance : AddCommGroup (DirectSum ι β) :=
  inferInstanceAs (AddCommGroup (Π₀ i, β i))

@[simp]
theorem sub_apply (g₁ g₂ : ⨁ i, β i) (i : ι) : (g₁ - g₂) i = g₁ i - g₂ i :=
  rfl


@[simp]
theorem zero_apply (i : ι) : (0 : ⨁ i, β i) i = 0 :=
  rfl


@[simp]
theorem add_apply (g₁ g₂ : ⨁ i, β i) (i : ι) : (g₁ + g₂) i = g₁ i + g₂ i :=
  rfl


/-- `mk β s x` is the element of `⨁ i, β i` that is zero outside `s`
and has coefficient `x i` for `i` in `s`. -/
def mk (s : Finset ι) : (∀ i : (↑s : Set ι), β i.1) →+ ⨁ i, β i where
  toFun := DFinsupp.mk s
  map_add' _ _ := DFinsupp.mk_add
  map_zero' := DFinsupp.mk_zero


/-- `of i` is the natural inclusion map from `β i` to `⨁ i, β i`. -/
def of (i : ι) : β i →+ ⨁ i, β i :=
  DFinsupp.singleAddHom β i


@[simp]
theorem of_eq_same (i : ι) (x : β i) : (of _ i x) i = x :=
  DFinsupp.single_eq_same


theorem of_eq_of_ne (i j : ι) (x : β i) (h : i ≠ j) : (of _ i x) j = 0 :=
  DFinsupp.single_eq_of_ne h


lemma of_apply {i : ι} (j : ι) (x : β i) : of β i x j = if h : i = j then Eq.recOn h x else 0 :=
  DFinsupp.single_apply


theorem mk_apply_of_mem {s : Finset ι} {f : ∀ i : (↑s : Set ι), β i.val} {n : ι} (hn : n ∈ s) :
    mk β s f n = f ⟨n, hn⟩ := by
  /-
    ι : Type v
    β : ι → Type w
    inst✝¹ : (i : ι) → AddCommMonoid (β i)
    inst✝ : DecidableEq ι
    s : Finset ι
    f : (i : ↑↑s) → β ↑i
    n : ι
    hn : Membership.mem s n
    ⊢ Eq (((DirectSum.mk β s) f) n) (f ⟨n, hn⟩)
  -/
  dsimp only [Finset.coe_sort_coe, mk, AddMonoidHom.coe_mk, ZeroHom.coe_mk, DFinsupp.mk_apply]
  /-
    ι : Type v
    β : ι → Type w
    inst✝¹ : (i : ι) → AddCommMonoid (β i)
    inst✝ : DecidableEq ι
    s : Finset ι
    f : (i : ↑↑s) → β ↑i
    n : ι
    hn : Membership.mem s n
    ⊢ Eq (dite (Membership.mem s n) (fun H => f ⟨n, H⟩) fun H => 0) (f ⟨n, hn⟩)
  -/
  rw [dif_pos hn]
  /-
    🎉 no goals
  -/


theorem mk_apply_of_not_mem {s : Finset ι} {f : ∀ i : (↑s : Set ι), β i.val} {n : ι} (hn : n ∉ s) :
    mk β s f n = 0 := by
  /-
    ι : Type v
    β : ι → Type w
    inst✝¹ : (i : ι) → AddCommMonoid (β i)
    inst✝ : DecidableEq ι
    s : Finset ι
    f : (i : ↑↑s) → β ↑i
    n : ι
    hn : Not (Membership.mem s n)
    ⊢ Eq (((DirectSum.mk β s) f) n) 0
  -/
  dsimp only [Finset.coe_sort_coe, mk, AddMonoidHom.coe_mk, ZeroHom.coe_mk, DFinsupp.mk_apply]
  /-
    ι : Type v
    β : ι → Type w
    inst✝¹ : (i : ι) → AddCommMonoid (β i)
    inst✝ : DecidableEq ι
    s : Finset ι
    f : (i : ↑↑s) → β ↑i
    n : ι
    hn : Not (Membership.mem s n)
    ⊢ Eq (dite (Membership.mem s n) (fun H => f ⟨n, H⟩) fun H => 0) 0
  -/
  rw [dif_neg hn]
  /-
    🎉 no goals
  -/


@[simp]
theorem support_zero [∀ (i : ι) (x : β i), Decidable (x ≠ 0)] : (0 : ⨁ i, β i).support = ∅ :=
  DFinsupp.support_zero


@[simp]
theorem support_of [∀ (i : ι) (x : β i), Decidable (x ≠ 0)] (i : ι) (x : β i) (h : x ≠ 0) :
    (of _ i x).support = {i} :=
  DFinsupp.support_single_ne_zero h


theorem support_of_subset [∀ (i : ι) (x : β i), Decidable (x ≠ 0)] {i : ι} {b : β i} :
    (of _ i b).support ⊆ {i} :=
  DFinsupp.support_single_subset


theorem sum_support_of [∀ (i : ι) (x : β i), Decidable (x ≠ 0)] (x : ⨁ i, β i) :
    (∑ i ∈ x.support, of β i (x i)) = x :=
  DFinsupp.sum_single


theorem sum_univ_of [Fintype ι] (x : ⨁ i, β i) :
    ∑ i ∈ Finset.univ, of β i (x i) = x := by
  /-
    ι : Type v
    β : ι → Type w
    inst✝² : (i : ι) → AddCommMonoid (β i)
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    x : DirectSum ι fun i => β i
    ⊢ Eq (Finset.univ.sum fun i => (DirectSum.of β i) (x i)) x
  -/
  apply DFinsupp.ext (fun i ↦ ?_)
  /-
    ι : Type v
    β : ι → Type w
    inst✝² : (i : ι) → AddCommMonoid (β i)
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    x : DirectSum ι fun i => β i
    i : ι
    ⊢ Eq ((Finset.univ.sum fun i => (DirectSum.of β i) (x i)) i) (x i)
  -/
  rw [DFinsupp.finset_sum_apply]
  /-
    ι : Type v
    β : ι → Type w
    inst✝² : (i : ι) → AddCommMonoid (β i)
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    x : DirectSum ι fun i => β i
    i : ι
    ⊢ Eq (Finset.univ.sum fun a => ((DirectSum.of β a) (x a)) i) (x i)
  -/
  simp [of_apply]
  /-
    🎉 no goals
  -/


theorem mk_injective (s : Finset ι) : Function.Injective (mk β s) :=
  DFinsupp.mk_injective s


theorem of_injective (i : ι) : Function.Injective (of β i) :=
  DFinsupp.single_injective


@[elab_as_elim]
protected theorem induction_on {C : (⨁ i, β i) → Prop} (x : ⨁ i, β i) (H_zero : C 0)
    (H_basic : ∀ (i : ι) (x : β i), C (of β i x))
    (H_plus : ∀ x y, C x → C y → C (x + y)) : C x := by
  /-
    ι : Type v
    β : ι → Type w
    inst✝¹ : (i : ι) → AddCommMonoid (β i)
    inst✝ : DecidableEq ι
    C : (DirectSum ι fun i => β i) → Prop
    x : DirectSum ι fun i => β i
    H_zero : C 0
    H_basic : ∀ (i : ι) (x : β i), C ((DirectSum.of β i) x)
    H_plus : ∀ (x y : DirectSum ι fun i => β i), C x → C y → C (HAdd.hAdd x y)
    ⊢ C x
  -/
  apply DFinsupp.induction x H_zero
  /-
    ι : Type v
    β : ι → Type w
    inst✝¹ : (i : ι) → AddCommMonoid (β i)
    inst✝ : DecidableEq ι
    C : (DirectSum ι fun i => β i) → Prop
    x : DirectSum ι fun i => β i
    H_zero : C 0
    H_basic : ∀ (i : ι) (x : β i), C ((DirectSum.of β i) x)
    H_plus : ∀ (x y : DirectSum ι fun i => β i), C x → C y → C (HAdd.hAdd x y)
    ⊢ ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → C f → …
  -/
  intro i b f h1 h2 ih
  /-
    ι : Type v
    β : ι → Type w
    inst✝¹ : (i : ι) → AddCommMonoid (β i)
    inst✝ : DecidableEq ι
    C : (DirectSum ι fun i => β i) → Prop
    x : DirectSum ι fun i => β i
    H_zero : C 0
    H_basic : ∀ (i : ι) (x : β i), C ((DirectSum.of β i) x)
    H_plus : ∀ (x y : DirectSum ι fun i => β i), C x → C y → C (HAdd.hAdd x y)
    i : ι
    b : β i
    f : DFinsupp fun i => β i
    h1 : Eq (f i) 0
    h2 : Ne b 0
    ih : C f
    ⊢ C (HAdd.hAdd (DFinsupp.single i b) f)
  -/
  solve_by_elim
  /-
    🎉 no goals
  -/


/-- If two additive homomorphisms from `⨁ i, β i` are equal on each `of β i y`,
then they are equal. -/
theorem addHom_ext {γ : Type*} [AddMonoid γ] ⦃f g : (⨁ i, β i) →+ γ⦄
    (H : ∀ (i : ι) (y : β i), f (of _ i y) = g (of _ i y)) : f = g :=
  DFinsupp.addHom_ext H


/-- If two additive homomorphisms from `⨁ i, β i` are equal on each `of β i y`,
then they are equal.

See note [partially-applied ext lemmas]. -/
@[ext high]
theorem addHom_ext' {γ : Type*} [AddMonoid γ] ⦃f g : (⨁ i, β i) →+ γ⦄
    (H : ∀ i : ι, f.comp (of _ i) = g.comp (of _ i)) : f = g :=
  addHom_ext fun i => DFunLike.congr_fun <| H i


/-- `toAddMonoid φ` is the natural homomorphism from `⨁ i, β i` to `γ`
induced by a family `φ` of homomorphisms `β i → γ`. -/
def toAddMonoid : (⨁ i, β i) →+ γ :=
  DFinsupp.liftAddHom (β := β) φ


@[simp]
theorem toAddMonoid_of (i) (x : β i) : toAddMonoid φ (of β i x) = φ i x :=
  DFinsupp.liftAddHom_apply_single φ i x


theorem toAddMonoid.unique (f : ⨁ i, β i) : ψ f = toAddMonoid (fun i => ψ.comp (of β i)) f := by
  /-
    ι : Type v
    β : ι → Type w
    inst✝² : (i : ι) → AddCommMonoid (β i)
    inst✝¹ : DecidableEq ι
    γ : Type u₁
    inst✝ : AddCommMonoid γ
    ψ : AddMonoidHom (DirectSum ι fun i => β i) γ
    f : DirectSum ι fun i => β i
    ⊢ Eq (ψ f) ((DirectSum.toAddMonoid fun i => ψ.comp (DirectSum.of β i)) f)
  -/
  congr
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` applies addHom_ext' here, which isn't what we want.
  /-
    case e_a
    ι : Type v
    β : ι → Type w
    inst✝² : (i : ι) → AddCommMonoid (β i)
    inst✝¹ : DecidableEq ι
    γ : Type u₁
    inst✝ : AddCommMonoid γ
    ψ : AddMonoidHom (DirectSum ι fun i => β i) γ
    f : DirectSum ι fun i => β i
    ⊢ Eq ψ (DirectSum.toAddMonoid fun i => ψ.comp (DirectSum.of β i))
  -/
  apply DFinsupp.addHom_ext'
  /-
    case e_a.H
    ι : Type v
    β : ι → Type w
    inst✝² : (i : ι) → AddCommMonoid (β i)
    inst✝¹ : DecidableEq ι
    γ : Type u₁
    inst✝ : AddCommMonoid γ
    ψ : AddMonoidHom (DirectSum ι fun i => β i) γ
    f : DirectSum ι fun i => β i
    ⊢ ∀ (x : ι), Eq (ψ.comp (DFinsupp.singleAddHom (fun i => β i) x)) ((DirectSum. …
  -/
  simp [toAddMonoid, of]
  /-
    🎉 no goals
  -/


lemma toAddMonoid_injective : Injective (toAddMonoid : (∀ i, β i →+ γ) → (⨁ i, β i) →+ γ) :=
  DFinsupp.liftAddHom.injective


@[simp] lemma toAddMonoid_inj {f g : ∀ i, β i →+ γ} : toAddMonoid f = toAddMonoid g ↔ f = g :=
  toAddMonoid_injective.eq_iff


/-- `fromAddMonoid φ` is the natural homomorphism from `γ` to `⨁ i, β i`
induced by a family `φ` of homomorphisms `γ → β i`.

Note that this is not an isomorphism. Not every homomorphism `γ →+ ⨁ i, β i` arises in this way. -/
def fromAddMonoid : (⨁ i, γ →+ β i) →+ γ →+ ⨁ i, β i :=
  toAddMonoid fun i => AddMonoidHom.compHom (of β i)


@[simp]
theorem fromAddMonoid_of (i : ι) (f : γ →+ β i) : fromAddMonoid (of _ i f) = (of _ i).comp f := by
  /-
    ι : Type v
    β : ι → Type w
    inst✝² : (i : ι) → AddCommMonoid (β i)
    inst✝¹ : DecidableEq ι
    γ : Type u₁
    inst✝ : AddCommMonoid γ
    i : ι
    f : AddMonoidHom γ (β i)
    ⊢ Eq (DirectSum.fromAddMonoid ((DirectSum.of (fun i => AddMonoidHom γ (β i)) i …
  -/
  rw [fromAddMonoid, toAddMonoid_of]
  /-
    ι : Type v
    β : ι → Type w
    inst✝² : (i : ι) → AddCommMonoid (β i)
    inst✝¹ : DecidableEq ι
    γ : Type u₁
    inst✝ : AddCommMonoid γ
    i : ι
    f : AddMonoidHom γ (β i)
    ⊢ Eq ((AddMonoidHom.compHom (DirectSum.of β i)) f) ((DirectSum.of β i).comp f)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem fromAddMonoid_of_apply (i : ι) (f : γ →+ β i) (x : γ) :
    fromAddMonoid (of _ i f) x = of _ i (f x) := by
      /-
        ι : Type v
        β : ι → Type w
        inst✝² : (i : ι) → AddCommMonoid (β i)
        inst✝¹ : DecidableEq ι
        γ : Type u₁
        inst✝ : AddCommMonoid γ
        i : ι
        f : AddMonoidHom γ (β i)
        x : γ
        ⊢ Eq ((DirectSum.fromAddMonoid ((DirectSum.of (fun i => AddMonoidHom γ (β i))  …
      -/
      rw [fromAddMonoid_of, AddMonoidHom.coe_comp, Function.comp]
      /-
        🎉 no goals
      -/


/-- `setToSet β S T h` is the natural homomorphism `⨁ (i : S), β i → ⨁ (i : T), β i`,
where `h : S ⊆ T`. -/
def setToSet (S T : Set ι) (H : S ⊆ T) : (⨁ i : S, β i) →+ ⨁ i : T, β i :=
  toAddMonoid fun i => of (fun i : Subtype T => β i) ⟨↑i, H i.2⟩


instance unique [∀ i, Subsingleton (β i)] : Unique (⨁ i, β i) :=
  DFinsupp.unique


/-- A direct sum over an empty type is trivial. -/
instance uniqueOfIsEmpty [IsEmpty ι] : Unique (⨁ i, β i) :=
  DFinsupp.uniqueOfIsEmpty


/-- The natural equivalence between `⨁ _ : ι, M` and `M` when `Unique ι`. -/
protected def id (M : Type v) (ι : Type* := PUnit) [AddCommMonoid M] [Unique ι] :
    (⨁ _ : ι, M) ≃+ M :=
  {
    DirectSum.toAddMonoid fun _ =>
      AddMonoidHom.id
        M with
    toFun := DirectSum.toAddMonoid fun _ => AddMonoidHom.id M
    invFun := of (fun _ => M) default
    left_inv := fun x =>
                                   /-
                                     ι✝ : Type v
                                     β : ι✝ → Type w
                                     inst✝² : (i : ι✝) → AddCommMonoid (β i)
                                     M : Type v
                                     ι : optParam (Type u_1) PUnit.{u_1 + 1}
                                     inst✝¹ : AddCommMonoid M
                                     inst✝ : Unique ι
                                     x : DirectSum ι fun x => M
                                     ⊢ Eq ((DirectSum.of (fun x => M) Inhabited.default) ((DirectSum.toAddMonoid fu …
                                   -/
      DirectSum.induction_on x (by rw [AddMonoidHom.map_zero, AddMonoidHom.map_zero])
                                   /-
                                     🎉 no goals
                                   -/
                       /-
                         ι✝ : Type v
                         β : ι✝ → Type w
                         inst✝² : (i : ι✝) → AddCommMonoid (β i)
                         M : Type v
                         ι : optParam (Type u_1) PUnit.{u_1 + 1}
                         inst✝¹ : AddCommMonoid M
                         inst✝ : Unique ι
                         x✝ : DirectSum ι fun x => M
                         p : ι
                         x : M
                         ⊢ Eq ((DirectSum.of (fun x => M) Inhabited.default) ((DirectSum.toAddMonoid fu …
                       -/
        (fun p x => by rw [Unique.default_eq p, toAddMonoid_of]; rfl) fun x y ihx ihy => by
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
        /-
          ι✝ : Type v
          β : ι✝ → Type w
          inst✝² : (i : ι✝) → AddCommMonoid (β i)
          M : Type v
          ι : optParam (Type u_1) PUnit.{u_1 + 1}
          inst✝¹ : AddCommMonoid M
          inst✝ : Unique ι
          x✝ x y : DirectSum ι fun i => M
          ihx : Eq ((DirectSum.of (fun x => M) Inhabited.default) ((DirectSum.toAddMonoi …
          ihy : Eq ((DirectSum.of (fun x => M) Inhabited.default) ((DirectSum.toAddMonoi …
          ⊢ Eq ((DirectSum.of (fun x => M) Inhabited.default) ((DirectSum.toAddMonoid fu …
        -/
        rw [AddMonoidHom.map_add, AddMonoidHom.map_add, ihx, ihy]
        /-
          🎉 no goals
        -/
    right_inv := fun _ => toAddMonoid_of _ _ _ }


/-- Reindexing terms of a direct sum. -/
def equivCongrLeft (h : ι ≃ κ) : (⨁ i, β i) ≃+ ⨁ k, β (h.symm k) :=
  { DFinsupp.equivCongrLeft h with map_add' := DFinsupp.comapDomain'_add _ h.right_inv}


@[simp]
theorem equivCongrLeft_apply (h : ι ≃ κ) (f : ⨁ i, β i) (k : κ) :
    equivCongrLeft h f k = f (h.symm k) := by
  /-
    ι : Type v
    β : ι → Type w
    inst✝ : (i : ι) → AddCommMonoid (β i)
    κ : Type u_1
    h : Equiv ι κ
    f : DirectSum ι fun i => β i
    k : κ
    ⊢ Eq (((DirectSum.equivCongrLeft h) f) k) (f (h.symm k))
  -/
  exact DFinsupp.comapDomain'_apply _ h.right_inv _ _
  /-
    🎉 no goals
  -/


/-- Isomorphism obtained by separating the term of index `none` of a direct sum over `Option ι`. -/
@[simps!]
noncomputable def addEquivProdDirectSum : (⨁ i, α i) ≃+ α none × ⨁ i, α (some i) :=
  { DFinsupp.equivProdDFinsupp with map_add' := DFinsupp.equivProdDFinsupp_add }


/-- The natural map between `⨁ (i : Σ i, α i), δ i.1 i.2` and `⨁ i (j : α i), δ i j`. -/
def sigmaCurry : (⨁ i : Σ _i, _, δ i.1 i.2) →+ ⨁ (i) (j), δ i j where
  toFun := DFinsupp.sigmaCurry (δ := δ)
  map_zero' := DFinsupp.sigmaCurry_zero
  map_add' f g := DFinsupp.sigmaCurry_add f g


@[simp]
theorem sigmaCurry_apply (f : ⨁ i : Σ _i, _, δ i.1 i.2) (i : ι) (j : α i) :
    sigmaCurry f i j = f ⟨i, j⟩ :=
  DFinsupp.sigmaCurry_apply (δ := δ) _ i j


/-- The natural map between `⨁ i (j : α i), δ i j` and `Π₀ (i : Σ i, α i), δ i.1 i.2`, inverse of
`curry`. -/
def sigmaUncurry : (⨁ (i) (j), δ i j) →+ ⨁ i : Σ _i, _, δ i.1 i.2 where
  toFun := DFinsupp.sigmaUncurry
  map_zero' := DFinsupp.sigmaUncurry_zero
  map_add' := DFinsupp.sigmaUncurry_add


@[simp]
theorem sigmaUncurry_apply (f : ⨁ (i) (j), δ i j) (i : ι) (j : α i) :
    sigmaUncurry f ⟨i, j⟩ = f i j :=
  DFinsupp.sigmaUncurry_apply f i j


/-- The natural map between `⨁ (i : Σ i, α i), δ i.1 i.2` and `⨁ i (j : α i), δ i j`. -/
def sigmaCurryEquiv : (⨁ i : Σ _i, _, δ i.1 i.2) ≃+ ⨁ (i) (j), δ i j :=
  { sigmaCurry, DFinsupp.sigmaCurryEquiv with }


/-- The canonical embedding from `⨁ i, A i` to `M` where `A` is a collection of `AddSubmonoid M`
indexed by `ι`.

When `S = Submodule _ M`, this is available as a `LinearMap`, `DirectSum.coe_linearMap`. -/
protected def coeAddMonoidHom {M S : Type*} [DecidableEq ι] [AddCommMonoid M] [SetLike S M]
    [AddSubmonoidClass S M] (A : ι → S) : (⨁ i, A i) →+ M :=
  toAddMonoid fun i => AddSubmonoidClass.subtype (A i)


theorem coeAddMonoidHom_eq_dfinsupp_sum [DecidableEq ι]
    {M S : Type*} [DecidableEq M] [AddCommMonoid M]
    [SetLike S M] [AddSubmonoidClass S M] (A : ι → S) (x : DirectSum ι fun i => A i) :
    DirectSum.coeAddMonoidHom A x = DFinsupp.sum x fun i => (fun x : A i => ↑x) := by
  simp only [DirectSum.coeAddMonoidHom, toAddMonoid, DFinsupp.liftAddHom, AddEquiv.coe_mk,
    Equiv.coe_fn_mk]
  /-
    ι : Type v
    inst✝⁴ : DecidableEq ι
    M : Type u_1
    S : Type u_2
    inst✝³ : DecidableEq M
    inst✝² : AddCommMonoid M
    inst✝¹ : SetLike S M
    inst✝ : AddSubmonoidClass S M
    A : ι → S
    x : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
    ⊢ Eq ((DFinsupp.sumAddHom fun i => AddSubmonoidClass.subtype (A i)) x) (DFinsu …
  -/
  exact DFinsupp.sumAddHom_apply _ x
  /-
    🎉 no goals
  -/


@[simp]
theorem coeAddMonoidHom_of {M S : Type*} [DecidableEq ι] [AddCommMonoid M] [SetLike S M]
    [AddSubmonoidClass S M] (A : ι → S) (i : ι) (x : A i) :
    DirectSum.coeAddMonoidHom A (of (fun i => A i) i x) = x :=
  toAddMonoid_of _ _ _


theorem coe_of_apply {M S : Type*} [DecidableEq ι] [AddCommMonoid M] [SetLike S M]
    [AddSubmonoidClass S M] {A : ι → S} (i j : ι) (x : A i) :
    (of (fun i ↦ {x // x ∈ A i}) i x j : M) = if i = j then x else 0 := by
  /-
    ι : Type v
    M : Type u_1
    S : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCommMonoid M
    inst✝¹ : SetLike S M
    inst✝ : AddSubmonoidClass S M
    A : ι → S
    i j : ι
    x : Subtype fun x => Membership.mem (A i) x
    ⊢ Eq ↑(((DirectSum.of (fun i => Subtype fun x => Membership.mem (A i) x) i) x) …
  -/
  obtain rfl | h := Decidable.eq_or_ne i j
    /-
      case inl
      ι : Type v
      M : Type u_1
      S : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid M
      inst✝¹ : SetLike S M
      inst✝ : AddSubmonoidClass S M
      A : ι → S
      i : ι
      x : Subtype fun x => Membership.mem (A i) x
      ⊢ Eq ↑(((DirectSum.of (fun i => Subtype fun x => Membership.mem (A i) x) i) x) …
    -/
  · rw [DirectSum.of_eq_same, if_pos rfl]
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type v
      M : Type u_1
      S : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid M
      inst✝¹ : SetLike S M
      inst✝ : AddSubmonoidClass S M
      A : ι → S
      i j : ι
      x : Subtype fun x => Membership.mem (A i) x
      h : Ne i j
      ⊢ Eq ↑(((DirectSum.of (fun i => Subtype fun x => Membership.mem (A i) x) i) x) …
    -/
  · rw [DirectSum.of_eq_of_ne _ _ _ h, if_neg h, ZeroMemClass.coe_zero, ZeroMemClass.coe_zero]
    /-
      🎉 no goals
    -/


/-- The `DirectSum` formed by a collection of additive submonoids (or subgroups, or submodules) of
`M` is said to be internal if the canonical map `(⨁ i, A i) →+ M` is bijective.

For the alternate statement in terms of independence and spanning, see
`DirectSum.subgroup_isInternal_iff_iSupIndep_and_supr_eq_top` and
`DirectSum.isInternal_submodule_iff_iSupIndep_and_iSup_eq_top`. -/
def IsInternal {M S : Type*} [DecidableEq ι] [AddCommMonoid M] [SetLike S M]
    [AddSubmonoidClass S M] (A : ι → S) : Prop :=
  Function.Bijective (DirectSum.coeAddMonoidHom A)


theorem IsInternal.addSubmonoid_iSup_eq_top {M : Type*} [DecidableEq ι] [AddCommMonoid M]
    (A : ι → AddSubmonoid M) (h : IsInternal A) : iSup A = ⊤ := by
  /-
    ι : Type v
    M : Type u_1
    inst✝¹ : DecidableEq ι
    inst✝ : AddCommMonoid M
    A : ι → AddSubmonoid M
    h : DirectSum.IsInternal A
    ⊢ Eq (iSup A) Top.top
  -/
  rw [AddSubmonoid.iSup_eq_mrange_dfinsupp_sumAddHom, AddMonoidHom.mrange_eq_top]
  /-
    ι : Type v
    M : Type u_1
    inst✝¹ : DecidableEq ι
    inst✝ : AddCommMonoid M
    A : ι → AddSubmonoid M
    h : DirectSum.IsInternal A
    ⊢ Function.Surjective ⇑(DFinsupp.sumAddHom fun i => (A i).subtype)
  -/
  exact Function.Bijective.surjective h
  /-
    🎉 no goals
  -/


theorem support_subset [DecidableEq ι] [DecidableEq M] (A : ι → S) (x : DirectSum ι fun i => A i) :
    (Function.support fun i => (x i : M)) ⊆ ↑(DFinsupp.support x) := by
  /-
    ι : Type v
    M : Type u_1
    S : Type u_2
    inst✝⁴ : AddCommMonoid M
    inst✝³ : SetLike S M
    inst✝² : AddSubmonoidClass S M
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq M
    A : ι → S
    x : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
    ⊢ HasSubset.Subset (Function.support fun i => ↑(x i)) ↑(DFinsupp.support x)
  -/
  intro m
  simp only [Function.mem_support, Finset.mem_coe, DFinsupp.mem_support_toFun, not_imp_not,
    ZeroMemClass.coe_eq_zero, imp_self]


theorem finite_support (A : ι → S) (x : DirectSum ι fun i => A i) :
    (Function.support fun i => (x i : M)).Finite := by
  classical
  exact (DFinsupp.support x).finite_toSet.subset (DirectSum.support_subset _ x)


/-- The canonical isomorphism of a finite direct sum of additive commutative monoids
and the corresponding finite product. -/
def DirectSum.addEquivProd {ι : Type*} [Fintype ι] (G : ι → Type*) [(i : ι) → AddCommMonoid (G i)] :
    DirectSum ι G ≃+ ((i : ι) → G i) :=
  ⟨DFinsupp.equivFunOnFintype, fun g h ↦ funext fun _ ↦ by
    simp only [DFinsupp.equivFunOnFintype, Equiv.toFun_as_coe, Equiv.coe_fn_mk, add_apply,
      Pi.add_apply]⟩

