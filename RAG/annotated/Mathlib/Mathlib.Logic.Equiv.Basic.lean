/-- `PProd α β` is equivalent to `α × β` -/
@[simps apply symm_apply]
def pprodEquivProd {α β} : PProd α β ≃ α × β where
  toFun x := (x.1, x.2)
  invFun x := ⟨x.1, x.2⟩
  left_inv := fun _ => rfl
  right_inv := fun _ => rfl


/-- Product of two equivalences, in terms of `PProd`. If `α ≃ β` and `γ ≃ δ`, then
`PProd α γ ≃ PProd β δ`. -/
-- Porting note: in Lean 3 this had `@[congr]`
@[simps apply]
def pprodCongr (e₁ : α ≃ β) (e₂ : γ ≃ δ) : PProd α γ ≃ PProd β δ where
  toFun x := ⟨e₁ x.1, e₂ x.2⟩
  invFun x := ⟨e₁.symm x.1, e₂.symm x.2⟩
                               /-
                                 α : Sort u_1
                                 α₁ : Sort u_2
                                 α₂ : Sort u_3
                                 β : Sort u_4
                                 β₁ : Sort u_5
                                 β₂ : Sort u_6
                                 γ : Sort u_7
                                 δ : Sort u_8
                                 e₁ : Equiv α β
                                 e₂ : Equiv γ δ
                                 x✝ : PProd α γ
                                 x : α
                                 y : γ
                                 ⊢ Eq ((fun x => { fst := e₁.symm x.fst, snd := e₂.symm x.snd }) ((fun x => { f …
                               -/
  left_inv := fun ⟨x, y⟩ => by simp
                               /-
                                 🎉 no goals
                               -/
                                /-
                                  α : Sort u_1
                                  α₁ : Sort u_2
                                  α₂ : Sort u_3
                                  β : Sort u_4
                                  β₁ : Sort u_5
                                  β₂ : Sort u_6
                                  γ : Sort u_7
                                  δ : Sort u_8
                                  e₁ : Equiv α β
                                  e₂ : Equiv γ δ
                                  x✝ : PProd β δ
                                  x : β
                                  y : δ
                                  ⊢ Eq ((fun x => { fst := e₁ x.fst, snd := e₂ x.snd }) ((fun x => { fst := e₁.s …
                                -/
  right_inv := fun ⟨x, y⟩ => by simp
                                /-
                                  🎉 no goals
                                -/


/-- Combine two equivalences using `PProd` in the domain and `Prod` in the codomain. -/
@[simps! apply symm_apply]
def pprodProd {α₂ β₂} (ea : α₁ ≃ α₂) (eb : β₁ ≃ β₂) :
    PProd α₁ β₁ ≃ α₂ × β₂ :=
  (ea.pprodCongr eb).trans pprodEquivProd


/-- Combine two equivalences using `PProd` in the codomain and `Prod` in the domain. -/
@[simps! apply symm_apply]
def prodPProd {α₁ β₁} (ea : α₁ ≃ α₂) (eb : β₁ ≃ β₂) :
    α₁ × β₁ ≃ PProd α₂ β₂ :=
  (ea.symm.pprodProd eb.symm).symm


/-- `PProd α β` is equivalent to `PLift α × PLift β` -/
@[simps! apply symm_apply]
def pprodEquivProdPLift : PProd α β ≃ PLift α × PLift β :=
  Equiv.plift.symm.pprodProd Equiv.plift.symm


/-- Product of two equivalences. If `α₁ ≃ α₂` and `β₁ ≃ β₂`, then `α₁ × β₁ ≃ α₂ × β₂`. This is
`Prod.map` as an equivalence. -/
-- Porting note: in Lean 3 there was also a @[congr] tag
@[simps (config := .asFn) apply]
def prodCongr {α₁ α₂ β₁ β₂} (e₁ : α₁ ≃ α₂) (e₂ : β₁ ≃ β₂) : α₁ × β₁ ≃ α₂ × β₂ :=
                                                              /-
                                                                α : Sort u_1
                                                                α₁✝ : Sort u_2
                                                                α₂✝ : Sort u_3
                                                                β : Sort u_4
                                                                β₁✝ : Sort u_5
                                                                β₂✝ : Sort u_6
                                                                γ : Sort u_7
                                                                δ : Sort u_8
                                                                α₁ : Type ?u.2571
                                                                α₂ : Type ?u.2573
                                                                β₁ : Type ?u.2570
                                                                β₂ : Type ?u.2572
                                                                e₁ : Equiv α₁ α₂
                                                                e₂ : Equiv β₁ β₂
                                                                x✝ : Prod α₁ β₁
                                                                a : α₁
                                                                b : β₁
                                                                ⊢ Eq (Prod.map (⇑e₁.symm) (⇑e₂.symm) (Prod.map ⇑e₁ ⇑e₂ { fst := a, snd := b }) …
                                                              -/
                                                              /-
                                                                🎉 no goals
                                                              -/
  ⟨Prod.map e₁ e₂, Prod.map e₁.symm e₂.symm, fun ⟨a, b⟩ => by simp, fun ⟨a, b⟩ => by simp⟩
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[simp]
theorem prodCongr_symm {α₁ α₂ β₁ β₂} (e₁ : α₁ ≃ α₂) (e₂ : β₁ ≃ β₂) :
    (prodCongr e₁ e₂).symm = prodCongr e₁.symm e₂.symm :=
  rfl


/-- Type product is commutative up to an equivalence: `α × β ≃ β × α`. This is `Prod.swap` as an
equivalence. -/
def prodComm (α β) : α × β ≃ β × α :=
  ⟨Prod.swap, Prod.swap, Prod.swap_swap, Prod.swap_swap⟩


@[simp]
theorem coe_prodComm (α β) : (⇑(prodComm α β) : α × β → β × α) = Prod.swap :=
  rfl


@[simp]
theorem prodComm_apply {α β} (x : α × β) : prodComm α β x = x.swap :=
  rfl


@[simp]
theorem prodComm_symm (α β) : (prodComm α β).symm = prodComm β α :=
  rfl


/-- Type product is associative up to an equivalence. -/
@[simps]
def prodAssoc (α β γ) : (α × β) × γ ≃ α × β × γ :=
  ⟨fun p => (p.1.1, p.1.2, p.2), fun p => ((p.1, p.2.1), p.2.2), fun ⟨⟨_, _⟩, _⟩ => rfl,
    fun ⟨_, ⟨_, _⟩⟩ => rfl⟩


/-- Four-way commutativity of `prod`. The name matches `mul_mul_mul_comm`. -/
@[simps apply]
def prodProdProdComm (α β γ δ) : (α × β) × γ × δ ≃ (α × γ) × β × δ where
  toFun abcd := ((abcd.1.1, abcd.2.1), (abcd.1.2, abcd.2.2))
  invFun acbd := ((acbd.1.1, acbd.2.1), (acbd.1.2, acbd.2.2))
  left_inv := fun ⟨⟨_a, _b⟩, ⟨_c, _d⟩⟩ => rfl
  right_inv := fun ⟨⟨_a, _c⟩, ⟨_b, _d⟩⟩ => rfl


@[simp]
theorem prodProdProdComm_symm (α β γ δ) :
    (prodProdProdComm α β γ δ).symm = prodProdProdComm α γ β δ :=
  rfl


/-- `γ`-valued functions on `α × β` are equivalent to functions `α → β → γ`. -/
@[simps (config := .asFn)]
def curry (α β γ) : (α × β → γ) ≃ (α → β → γ) where
  toFun := Function.curry
  invFun := uncurry
  left_inv := uncurry_curry
  right_inv := curry_uncurry


/-- `PUnit` is a right identity for type product up to an equivalence. -/
@[simps]
def prodPUnit (α) : α × PUnit ≃ α :=
  ⟨fun p => p.1, fun a => (a, PUnit.unit), fun ⟨_, PUnit.unit⟩ => rfl, fun _ => rfl⟩


/-- `PUnit` is a left identity for type product up to an equivalence. -/
@[simps!]
def punitProd (α) : PUnit × α ≃ α :=
  calc
    PUnit × α ≃ α × PUnit := prodComm _ _
    _ ≃ α := prodPUnit _


/-- `PUnit` is a right identity for dependent type product up to an equivalence. -/
@[simps]
def sigmaPUnit (α) : (_ : α) × PUnit ≃ α :=
  ⟨fun p => p.1, fun a => ⟨a, PUnit.unit⟩, fun ⟨_, PUnit.unit⟩ => rfl, fun _ => rfl⟩


/-- Any `Unique` type is a right identity for type product up to equivalence. -/
def prodUnique (α β) [Unique β] : α × β ≃ α :=
  ((Equiv.refl α).prodCongr <| equivPUnit.{_,1} β).trans <| prodPUnit α


@[simp]
theorem coe_prodUnique {α β} [Unique β] : (⇑(prodUnique α β) : α × β → α) = Prod.fst :=
  rfl


theorem prodUnique_apply {α β} [Unique β] (x : α × β) : prodUnique α β x = x.1 :=
  rfl


@[simp]
theorem prodUnique_symm_apply {α β} [Unique β] (x : α) : (prodUnique α β).symm x = (x, default) :=
  rfl


/-- Any `Unique` type is a left identity for type product up to equivalence. -/
def uniqueProd (α β) [Unique β] : β × α ≃ α :=
  ((equivPUnit.{_,1} β).prodCongr <| Equiv.refl α).trans <| punitProd α


@[simp]
theorem coe_uniqueProd {α β} [Unique β] : (⇑(uniqueProd α β) : β × α → α) = Prod.snd :=
  rfl


theorem uniqueProd_apply {α β} [Unique β] (x : β × α) : uniqueProd α β x = x.2 :=
  rfl


@[simp]
theorem uniqueProd_symm_apply {α β} [Unique β] (x : α) :
    (uniqueProd α β).symm x = (default, x) :=
  rfl


/-- Any family of `Unique` types is a right identity for dependent type product up to
equivalence. -/
def sigmaUnique (α) (β : α → Type*) [∀ a, Unique (β a)] : (a : α) × (β a) ≃ α :=
  (Equiv.sigmaCongrRight fun a ↦ equivPUnit.{_,1} (β a)).trans <| sigmaPUnit α


@[simp]
theorem coe_sigmaUnique {α} {β : α → Type*} [∀ a, Unique (β a)] :
    (⇑(sigmaUnique α β) : (a : α) × (β a) → α) = Sigma.fst :=
  rfl


theorem sigmaUnique_apply {α} {β : α → Type*} [∀ a, Unique (β a)] (x : (a : α) × β a) :
    sigmaUnique α β x = x.1 :=
  rfl


@[simp]
theorem sigmaUnique_symm_apply {α} {β : α → Type*} [∀ a, Unique (β a)] (x : α) :
    (sigmaUnique α β).symm x = ⟨x, default⟩ :=
  rfl


/-- `Empty` type is a right absorbing element for type product up to an equivalence. -/
def prodEmpty (α) : α × Empty ≃ Empty :=
  equivEmpty _


/-- `Empty` type is a left absorbing element for type product up to an equivalence. -/
def emptyProd (α) : Empty × α ≃ Empty :=
  equivEmpty _


/-- `PEmpty` type is a right absorbing element for type product up to an equivalence. -/
def prodPEmpty (α) : α × PEmpty ≃ PEmpty :=
  equivPEmpty _


/-- `PEmpty` type is a left absorbing element for type product up to an equivalence. -/
def pemptyProd (α) : PEmpty × α ≃ PEmpty :=
  equivPEmpty _


/-- `PSum` is equivalent to `Sum`. -/
def psumEquivSum (α β) : α ⊕' β ≃ α ⊕ β where
  toFun s := PSum.casesOn s inl inr
  invFun := Sum.elim PSum.inl PSum.inr
                   /-
                     α✝ : Sort u_1
                     α₁ : Sort u_2
                     α₂ : Sort u_3
                     β✝ : Sort u_4
                     β₁ : Sort u_5
                     β₂ : Sort u_6
                     γ : Sort u_7
                     δ : Sort u_8
                     α : Type ?u.10030
                     β : Type ?u.10029
                     s : PSum α β
                     ⊢ Eq (Sum.elim PSum.inl PSum.inr ((fun s => PSum.casesOn s Sum.inl Sum.inr) s) …
                   -/
                               /-
                                 🎉 no goals
                               -/
  left_inv s := by cases s <;> rfl
                               /-
                                 🎉 no goals
                               -/
                    /-
                      α✝ : Sort u_1
                      α₁ : Sort u_2
                      α₂ : Sort u_3
                      β✝ : Sort u_4
                      β₁ : Sort u_5
                      β₂ : Sort u_6
                      γ : Sort u_7
                      δ : Sort u_8
                      α : Type ?u.10030
                      β : Type ?u.10029
                      s : Sum α β
                      ⊢ Eq ((fun s => PSum.casesOn s Sum.inl Sum.inr) (Sum.elim PSum.inl PSum.inr s) …
                    -/
                                /-
                                  🎉 no goals
                                -/
  right_inv s := by cases s <;> rfl
                                /-
                                  🎉 no goals
                                -/


/-- If `α ≃ α'` and `β ≃ β'`, then `α ⊕ β ≃ α' ⊕ β'`. This is `Sum.map` as an equivalence. -/
@[simps apply]
def sumCongr {α₁ α₂ β₁ β₂} (ea : α₁ ≃ α₂) (eb : β₁ ≃ β₂) : α₁ ⊕ β₁ ≃ α₂ ⊕ β₂ :=
                                                       /-
                                                         α : Sort u_1
                                                         α₁✝ : Sort u_2
                                                         α₂✝ : Sort u_3
                                                         β : Sort u_4
                                                         β₁✝ : Sort u_5
                                                         β₂✝ : Sort u_6
                                                         γ : Sort u_7
                                                         δ : Sort u_8
                                                         α₁ : Type ?u.10336
                                                         α₂ : Type ?u.10338
                                                         β₁ : Type ?u.10335
                                                         β₂ : Type ?u.10337
                                                         ea : Equiv α₁ α₂
                                                         eb : Equiv β₁ β₂
                                                         x : Sum α₁ β₁
                                                         ⊢ Eq (Sum.map (⇑ea.symm) (⇑eb.symm) (Sum.map (⇑ea) (⇑eb) x)) x
                                                       -/
                                                       /-
                                                         🎉 no goals
                                                       -/
  ⟨Sum.map ea eb, Sum.map ea.symm eb.symm, fun x => by simp, fun x => by simp⟩
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- If `α ≃ α'` and `β ≃ β'`, then `α ⊕' β ≃ α' ⊕' β'`. -/
def psumCongr (e₁ : α ≃ β) (e₂ : γ ≃ δ) : α ⊕' γ ≃ β ⊕' δ where
  toFun x := PSum.casesOn x (PSum.inl ∘ e₁) (PSum.inr ∘ e₂)
  invFun x := PSum.casesOn x (PSum.inl ∘ e₁.symm) (PSum.inr ∘ e₂.symm)
                 /-
                   α : Sort u_1
                   α₁ : Sort u_2
                   α₂ : Sort u_3
                   β : Sort u_4
                   β₁ : Sort u_5
                   β₂ : Sort u_6
                   γ : Sort u_7
                   δ : Sort u_8
                   e₁ : Equiv α β
                   e₂ : Equiv γ δ
                   ⊢ Function.LeftInverse (fun x => PSum.casesOn x (Function.comp PSum.inl ⇑e₁.sy …
                 -/
                                    /-
                                      🎉 no goals
                                    -/
  left_inv := by rintro (x | x) <;> simp
                                    /-
                                      🎉 no goals
                                    -/
                  /-
                    α : Sort u_1
                    α₁ : Sort u_2
                    α₂ : Sort u_3
                    β : Sort u_4
                    β₁ : Sort u_5
                    β₂ : Sort u_6
                    γ : Sort u_7
                    δ : Sort u_8
                    e₁ : Equiv α β
                    e₂ : Equiv γ δ
                    ⊢ Function.RightInverse (fun x => PSum.casesOn x (Function.comp PSum.inl ⇑e₁.s …
                  -/
                                     /-
                                       🎉 no goals
                                     -/
  right_inv := by rintro (x | x) <;> simp
                                     /-
                                       🎉 no goals
                                     -/


/-- Combine two `Equiv`s using `PSum` in the domain and `Sum` in the codomain. -/
def psumSum {α₂ β₂} (ea : α₁ ≃ α₂) (eb : β₁ ≃ β₂) :
    α₁ ⊕' β₁ ≃ α₂ ⊕ β₂ :=
  (ea.psumCongr eb).trans (psumEquivSum _ _)


/-- Combine two `Equiv`s using `Sum` in the domain and `PSum` in the codomain. -/
def sumPSum {α₁ β₁} (ea : α₁ ≃ α₂) (eb : β₁ ≃ β₂) :
    α₁ ⊕ β₁ ≃ α₂ ⊕' β₂ :=
  (ea.symm.psumSum eb.symm).symm


@[simp]
theorem sumCongr_trans {α₁ α₂ β₁ β₂ γ₁ γ₂} (e : α₁ ≃ β₁) (f : α₂ ≃ β₂) (g : β₁ ≃ γ₁) (h : β₂ ≃ γ₂) :
    (Equiv.sumCongr e f).trans (Equiv.sumCongr g h) = Equiv.sumCongr (e.trans g) (f.trans h) := by
  /-
    α₁ : Type u_9
    α₂ : Type u_10
    β₁ : Type u_11
    β₂ : Type u_12
    γ₁ : Type u_13
    γ₂ : Type u_14
    e : Equiv α₁ β₁
    f : Equiv α₂ β₂
    g : Equiv β₁ γ₁
    h : Equiv β₂ γ₂
    ⊢ Eq ((e.sumCongr f).trans (g.sumCongr h)) ((e.trans g).sumCongr (f.trans h))
  -/
  ext i
  /-
    case H
    α₁ : Type u_9
    α₂ : Type u_10
    β₁ : Type u_11
    β₂ : Type u_12
    γ₁ : Type u_13
    γ₂ : Type u_14
    e : Equiv α₁ β₁
    f : Equiv α₂ β₂
    g : Equiv β₁ γ₁
    h : Equiv β₂ γ₂
    i : Sum α₁ α₂
    ⊢ Eq (((e.sumCongr f).trans (g.sumCongr h)) i) (((e.trans g).sumCongr (f.trans …
  -/
              /-
                🎉 no goals
              -/
  cases i <;> rfl
              /-
                🎉 no goals
              -/


@[simp]
theorem sumCongr_symm {α β γ δ} (e : α ≃ β) (f : γ ≃ δ) :
    (Equiv.sumCongr e f).symm = Equiv.sumCongr e.symm f.symm :=
  rfl


@[simp]
theorem sumCongr_refl {α β} :
    Equiv.sumCongr (Equiv.refl α) (Equiv.refl β) = Equiv.refl (α ⊕ β) := by
  /-
    α : Type u_9
    β : Type u_10
    ⊢ Eq ((Equiv.refl α).sumCongr (Equiv.refl β)) (Equiv.refl (Sum α β))
  -/
  ext i
  /-
    case H
    α : Type u_9
    β : Type u_10
    i : Sum α β
    ⊢ Eq (((Equiv.refl α).sumCongr (Equiv.refl β)) i) ((Equiv.refl (Sum α β)) i)
  -/
              /-
                🎉 no goals
              -/
  cases i <;> rfl
              /-
                🎉 no goals
              -/


/-- A subtype of a sum is equivalent to a sum of subtypes. -/
def subtypeSum {α β} {p : α ⊕ β → Prop} :
    {c // p c} ≃ {a // p (Sum.inl a)} ⊕ {b // p (Sum.inr b)} where
  toFun c := match h : c.1 with
    | Sum.inl a => Sum.inl ⟨a, h ▸ c.2⟩
    | Sum.inr b => Sum.inr ⟨b, h ▸ c.2⟩
  invFun c := match c with
    | Sum.inl a => ⟨Sum.inl a, a.2⟩
    | Sum.inr b => ⟨Sum.inr b, b.2⟩
                 /-
                   α✝ : Sort u_1
                   α₁ : Sort u_2
                   α₂ : Sort u_3
                   β✝ : Sort u_4
                   β₁ : Sort u_5
                   β₂ : Sort u_6
                   γ : Sort u_7
                   δ : Sort u_8
                   α : Type ?u.13643
                   β : Type ?u.13642
                   p : Sum α β → Prop
                   ⊢ Function.LeftInverse (fun c => Equiv.subtypeSum.match_2 (fun c => Subtype fu …
                 -/
                                       /-
                                         🎉 no goals
                                       -/
  left_inv := by rintro ⟨a | b, h⟩ <;> rfl
                                       /-
                                         🎉 no goals
                                       -/
                  /-
                    α✝ : Sort u_1
                    α₁ : Sort u_2
                    α₂ : Sort u_3
                    β✝ : Sort u_4
                    β₁ : Sort u_5
                    β₂ : Sort u_6
                    γ : Sort u_7
                    δ : Sort u_8
                    α : Type ?u.13643
                    β : Type ?u.13642
                    p : Sum α β → Prop
                    ⊢ Function.RightInverse (fun c => Equiv.subtypeSum.match_2 (fun c => Subtype f …
                  -/
                                     /-
                                       🎉 no goals
                                     -/
  right_inv := by rintro (a | b) <;> rfl
                                     /-
                                       🎉 no goals
                                     -/


/-- Combine a permutation of `α` and of `β` into a permutation of `α ⊕ β`. -/
abbrev sumCongr {α β} (ea : Equiv.Perm α) (eb : Equiv.Perm β) : Equiv.Perm (α ⊕ β) :=
  Equiv.sumCongr ea eb


@[simp]
theorem sumCongr_apply {α β} (ea : Equiv.Perm α) (eb : Equiv.Perm β) (x : α ⊕ β) :
    sumCongr ea eb x = Sum.map (⇑ea) (⇑eb) x :=
  Equiv.sumCongr_apply ea eb x

-- Porting note: it seems the general theorem about `Equiv` is now applied, so there's no need
-- to have this version also have `@[simp]`. Similarly for below.

theorem sumCongr_trans {α β} (e : Equiv.Perm α) (f : Equiv.Perm β) (g : Equiv.Perm α)
    (h : Equiv.Perm β) : (sumCongr e f).trans (sumCongr g h) = sumCongr (e.trans g) (f.trans h) :=
  Equiv.sumCongr_trans e f g h


theorem sumCongr_symm {α β} (e : Equiv.Perm α) (f : Equiv.Perm β) :
    (sumCongr e f).symm = sumCongr e.symm f.symm :=
  Equiv.sumCongr_symm e f


theorem sumCongr_refl {α β} : sumCongr (Equiv.refl α) (Equiv.refl β) = Equiv.refl (α ⊕ β) :=
  Equiv.sumCongr_refl


/-- `Bool` is equivalent the sum of two `PUnit`s. -/
def boolEquivPUnitSumPUnit : Bool ≃ PUnit.{u + 1} ⊕ PUnit.{v + 1} :=
  ⟨fun b => b.casesOn (inl PUnit.unit) (inr PUnit.unit) , Sum.elim (fun _ => false) fun _ => true,
                /-
                  α : Sort u_1
                  α₁ : Sort u_2
                  α₂ : Sort u_3
                  β : Sort u_4
                  β₁ : Sort u_5
                  β₂ : Sort u_6
                  γ : Sort u_7
                  δ : Sort u_8
                  b : Bool
                  ⊢ Eq (Sum.elim (fun x => Bool.false) (fun x => Bool.true) ((fun b => Bool.case …
                -/
                            /-
                              🎉 no goals
                            -/
                            /-
                              🎉 no goals
                            -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
    fun b => by cases b <;> rfl, fun s => by rcases s with (⟨⟨⟩⟩ | ⟨⟨⟩⟩) <;> rfl⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- Sum of types is commutative up to an equivalence. This is `Sum.swap` as an equivalence. -/
@[simps (config := .asFn) apply]
def sumComm (α β) : α ⊕ β ≃ β ⊕ α :=
  ⟨Sum.swap, Sum.swap, Sum.swap_swap, Sum.swap_swap⟩


@[simp]
theorem sumComm_symm (α β) : (sumComm α β).symm = sumComm β α :=
  rfl


/-- Sum of types is associative up to an equivalence. -/
def sumAssoc (α β γ) : (α ⊕ β) ⊕ γ ≃ α ⊕ (β ⊕ γ) :=
  ⟨Sum.elim (Sum.elim Sum.inl (Sum.inr ∘ Sum.inl)) (Sum.inr ∘ Sum.inr),
    Sum.elim (Sum.inl ∘ Sum.inl) <| Sum.elim (Sum.inl ∘ Sum.inr) Sum.inr,
         /-
           α✝ : Sort u_1
           α₁ : Sort u_2
           α₂ : Sort u_3
           β✝ : Sort u_4
           β₁ : Sort u_5
           β₂ : Sort u_6
           γ✝ : Sort u_7
           δ : Sort u_8
           α : Type ?u.15750
           β : Type ?u.15749
           γ : Type ?u.15747
           ⊢ Function.LeftInverse (Sum.elim (Function.comp Sum.inl Sum.inl) (Sum.elim (Fu …
         -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
      by rintro (⟨_ | _⟩ | _) <;> rfl, by
                                  /-
                                    🎉 no goals
                                  -/
    /-
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ✝ : Sort u_7
      δ : Sort u_8
      α : Type ?u.15750
      β : Type ?u.15749
      γ : Type ?u.15747
      ⊢ Function.RightInverse (Sum.elim (Function.comp Sum.inl Sum.inl) (Sum.elim (F …
    -/
                             /-
                               🎉 no goals
                             -/
                             /-
                               🎉 no goals
                             -/
    rintro (_ | ⟨_ | _⟩) <;> rfl⟩
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem sumAssoc_apply_inl_inl {α β γ} (a) : sumAssoc α β γ (inl (inl a)) = inl a :=
  rfl


@[simp]
theorem sumAssoc_apply_inl_inr {α β γ} (b) : sumAssoc α β γ (inl (inr b)) = inr (inl b) :=
  rfl


@[simp]
theorem sumAssoc_apply_inr {α β γ} (c) : sumAssoc α β γ (inr c) = inr (inr c) :=
  rfl


@[simp]
theorem sumAssoc_symm_apply_inl {α β γ} (a) : (sumAssoc α β γ).symm (inl a) = inl (inl a) :=
  rfl


@[simp]
theorem sumAssoc_symm_apply_inr_inl {α β γ} (b) :
    (sumAssoc α β γ).symm (inr (inl b)) = inl (inr b) :=
  rfl


@[simp]
theorem sumAssoc_symm_apply_inr_inr {α β γ} (c) : (sumAssoc α β γ).symm (inr (inr c)) = inr c :=
  rfl


/-- Four-way commutativity of `sum`. The name matches `add_add_add_comm`. -/
@[simps apply]
def sumSumSumComm (α β γ δ) : (α ⊕ β) ⊕ γ ⊕ δ ≃ (α ⊕ γ) ⊕ β ⊕ δ where
  toFun :=
    (sumAssoc (α ⊕ γ) β δ) ∘ (Sum.map (sumAssoc α γ β).symm (@id δ))
      ∘ (Sum.map (Sum.map (@id α) (sumComm β γ)) (@id δ))
      ∘ (Sum.map (sumAssoc α β γ) (@id δ))
      ∘ (sumAssoc (α ⊕ β) γ δ).symm
  invFun :=
    (sumAssoc (α ⊕ β) γ δ) ∘ (Sum.map (sumAssoc α β γ).symm (@id δ))
      ∘ (Sum.map (Sum.map (@id α) (sumComm β γ).symm) (@id δ))
      ∘ (Sum.map (sumAssoc α γ β) (@id δ))
      ∘ (sumAssoc (α ⊕ γ) β δ).symm
                   /-
                     α✝ : Sort u_1
                     α₁ : Sort u_2
                     α₂ : Sort u_3
                     β✝ : Sort u_4
                     β₁ : Sort u_5
                     β₂ : Sort u_6
                     γ✝ : Sort u_7
                     δ✝ : Sort u_8
                     α : Type ?u.17448
                     β : Type ?u.17447
                     γ : Type ?u.17450
                     δ : Type ?u.17449
                     x : Sum (Sum α β) (Sum γ δ)
                     ⊢ Eq (Function.comp (⇑(Equiv.sumAssoc (Sum α β) γ δ)) (Function.comp (Sum.map  …
                   -/
                                                         /-
                                                           🎉 no goals
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
  left_inv x := by rcases x with ((a | b) | (c | d)) <;> simp
                                                         /-
                                                           🎉 no goals
                                                         -/
                    /-
                      α✝ : Sort u_1
                      α₁ : Sort u_2
                      α₂ : Sort u_3
                      β✝ : Sort u_4
                      β₁ : Sort u_5
                      β₂ : Sort u_6
                      γ✝ : Sort u_7
                      δ✝ : Sort u_8
                      α : Type ?u.17448
                      β : Type ?u.17447
                      γ : Type ?u.17450
                      δ : Type ?u.17449
                      x : Sum (Sum α γ) (Sum β δ)
                      ⊢ Eq (Function.comp (⇑(Equiv.sumAssoc (Sum α γ) β δ)) (Function.comp (Sum.map  …
                    -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  right_inv x := by rcases x with ((a | c) | (b | d)) <;> simp
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
theorem sumSumSumComm_symm (α β γ δ) : (sumSumSumComm α β γ δ).symm = sumSumSumComm α γ β δ :=
  rfl


/-- Sum with `IsEmpty` is equivalent to the original type. -/
@[simps symm_apply]
def sumEmpty (α β) [IsEmpty β] : α ⊕ β ≃ α where
  toFun := Sum.elim id isEmptyElim
  invFun := inl
  left_inv s := by
    /-
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α : Type ?u.26909
      β : Type ?u.26908
      inst✝ : IsEmpty β
      s : Sum α β
      ⊢ Eq (Sum.inl (Sum.elim id (fun a => isEmptyElim a) s)) s
    -/
    rcases s with (_ | x)
      /-
        case inl
        α✝ : Sort u_1
        α₁ : Sort u_2
        α₂ : Sort u_3
        β✝ : Sort u_4
        β₁ : Sort u_5
        β₂ : Sort u_6
        γ : Sort u_7
        δ : Sort u_8
        α : Type ?u.26909
        β : Type ?u.26908
        inst✝ : IsEmpty β
        val✝ : α
        ⊢ Eq (Sum.inl (Sum.elim id (fun a => isEmptyElim a) (Sum.inl val✝))) (Sum.inl  …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case inr
        α✝ : Sort u_1
        α₁ : Sort u_2
        α₂ : Sort u_3
        β✝ : Sort u_4
        β₁ : Sort u_5
        β₂ : Sort u_6
        γ : Sort u_7
        δ : Sort u_8
        α : Type ?u.26909
        β : Type ?u.26908
        inst✝ : IsEmpty β
        x : β
        ⊢ Eq (Sum.inl (Sum.elim id (fun a => isEmptyElim a) (Sum.inr x))) (Sum.inr x)
      -/
    · exact isEmptyElim x
      /-
        🎉 no goals
      -/
  right_inv _ := rfl


@[simp]
theorem sumEmpty_apply_inl {α β} [IsEmpty β] (a : α) : sumEmpty α β (Sum.inl a) = a :=
  rfl


/-- The sum of `IsEmpty` with any type is equivalent to that type. -/
@[simps! symm_apply]
def emptySum (α β) [IsEmpty α] : α ⊕ β ≃ β :=
  (sumComm _ _).trans <| sumEmpty _ _


@[simp]
theorem emptySum_apply_inr {α β} [IsEmpty α] (b : β) : emptySum α β (Sum.inr b) = b :=
  rfl


/-- `Option α` is equivalent to `α ⊕ PUnit` -/
def optionEquivSumPUnit (α) : Option α ≃ α ⊕ PUnit :=
  ⟨fun o => o.elim (inr PUnit.unit) inl, fun s => s.elim some fun _ => none,
                /-
                  α✝ : Sort u_1
                  α₁ : Sort u_2
                  α₂ : Sort u_3
                  β : Sort u_4
                  β₁ : Sort u_5
                  β₂ : Sort u_6
                  γ : Sort u_7
                  δ : Sort u_8
                  α : Type ?u.27673
                  o : Option α
                  ⊢ Eq ((fun s => Sum.elim Option.some (fun x => Option.none) s) ((fun o => o.el …
                -/
                            /-
                              🎉 no goals
                            -/
    fun o => by cases o <;> rfl,
                            /-
                              🎉 no goals
                            -/
                /-
                  α✝ : Sort u_1
                  α₁ : Sort u_2
                  α₂ : Sort u_3
                  β : Sort u_4
                  β₁ : Sort u_5
                  β₂ : Sort u_6
                  γ : Sort u_7
                  δ : Sort u_8
                  α : Type ?u.27673
                  s : Sum α PUnit.{?u.27674 + 1}
                  ⊢ Eq ((fun o => o.elim (Sum.inr PUnit.unit) Sum.inl) ((fun s => Sum.elim Optio …
                -/
                                             /-
                                               🎉 no goals
                                             -/
    fun s => by rcases s with (_ | ⟨⟨⟩⟩) <;> rfl⟩
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem optionEquivSumPUnit_none {α} : optionEquivSumPUnit α none = Sum.inr PUnit.unit :=
  rfl


@[simp]
theorem optionEquivSumPUnit_some {α} (a) : optionEquivSumPUnit α (some a) = Sum.inl a :=
  rfl


@[simp]
theorem optionEquivSumPUnit_coe {α} (a : α) : optionEquivSumPUnit α a = Sum.inl a :=
  rfl


@[simp]
theorem optionEquivSumPUnit_symm_inl {α} (a) : (optionEquivSumPUnit α).symm (Sum.inl a) = a :=
  rfl


@[simp]
theorem optionEquivSumPUnit_symm_inr {α} (a) : (optionEquivSumPUnit α).symm (Sum.inr a) = none :=
  rfl


/-- The set of `x : Option α` such that `isSome x` is equivalent to `α`. -/
@[simps]
def optionIsSomeEquiv (α) : { x : Option α // x.isSome } ≃ α where
  toFun o := Option.get _ o.2
  invFun x := ⟨some x, rfl⟩
  left_inv _ := Subtype.eq <| Option.some_get _
  right_inv _ := Option.get_some _ _


/-- The product over `Option α` of `β a` is the binary product of the
product over `α` of `β (some α)` and `β none` -/
@[simps]
def piOptionEquivProd {α} {β : Option α → Type*} :
    (∀ a : Option α, β a) ≃ β none × ∀ a : α, β (some a) where
  toFun f := (f none, fun a => f (some a))
  invFun x a := Option.casesOn a x.fst x.snd
                                   /-
                                     α✝ : Sort u_1
                                     α₁ : Sort u_2
                                     α₂ : Sort u_3
                                     β✝ : Sort u_4
                                     β₁ : Sort u_5
                                     β₂ : Sort u_6
                                     γ : Sort u_7
                                     δ : Sort u_8
                                     α : Type ?u.29340
                                     β : Option α → Type u_9
                                     f : (a : Option α) → β a
                                     a : Option α
                                     ⊢ Eq ((fun x a => Option.casesOn a x.1 x.2) ((fun f => { fst := f Option.none, …
                                   -/
                                               /-
                                                 🎉 no goals
                                               -/
  left_inv f := funext fun a => by cases a <;> rfl
                                               /-
                                                 🎉 no goals
                                               -/
                    /-
                      α✝ : Sort u_1
                      α₁ : Sort u_2
                      α₂ : Sort u_3
                      β✝ : Sort u_4
                      β₁ : Sort u_5
                      β₂ : Sort u_6
                      γ : Sort u_7
                      δ : Sort u_8
                      α : Type ?u.29340
                      β : Option α → Type u_9
                      x : Prod (β Option.none) ((a : α) → β (Option.some a))
                      ⊢ Eq ((fun f => { fst := f Option.none, snd := fun a => f (Option.some a) }) ( …
                    -/
  right_inv x := by simp
                    /-
                      🎉 no goals
                    -/


/-- `α ⊕ β` is equivalent to a `Sigma`-type over `Bool`. Note that this definition assumes `α` and
`β` to be types from the same universe, so it cannot be used directly to transfer theorems about
sigma types to theorems about sum types. In many cases one can use `ULift` to work around this
difficulty. -/
def sumEquivSigmaBool (α β) : α ⊕ β ≃ Σ b : Bool, b.casesOn α β :=
  ⟨fun s => s.elim (fun x => ⟨false, x⟩) fun x => ⟨true, x⟩, fun s =>
    match s with
    | ⟨false, a⟩ => inl a
    | ⟨true, b⟩ => inr b,
                /-
                  α✝ : Sort u_1
                  α₁ : Sort u_2
                  α₂ : Sort u_3
                  β✝ : Sort u_4
                  β₁ : Sort u_5
                  β₂ : Sort u_6
                  γ : Sort u_7
                  δ : Sort u_8
                  α β : Type ?u.29732
                  s : Sum α β
                  ⊢ Eq ((fun s => Equiv.sumEquivSigmaBool.match_1 α β (fun s => Sum α β) s (fun  …
                -/
                            /-
                              🎉 no goals
                            -/
                            /-
                              🎉 no goals
                            -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    fun s => by cases s <;> rfl, fun s => by rcases s with ⟨_ | _, _⟩ <;> rfl⟩
                                                                          /-
                                                                            🎉 no goals
                                                                          -/

-- See also `Equiv.sigmaPreimageEquiv`.

/-- `sigmaFiberEquiv f` for `f : α → β` is the natural equivalence between
the type of all fibres of `f` and the total space `α`. -/
@[simps]
def sigmaFiberEquiv {α β : Type*} (f : α → β) : (Σ y : β, { x // f x = y }) ≃ α :=
  ⟨fun x => ↑x.2, fun x => ⟨f x, x, rfl⟩, fun ⟨_, _, rfl⟩ => rfl, fun _ => rfl⟩


/-- Inhabited types are equivalent to `Option β` for some `β` by identifying `default` with `none`.
-/
def sigmaEquivOptionOfInhabited (α : Type u) [Inhabited α] [DecidableEq α] :
    Σ β : Type u, α ≃ Option β where
  fst := {a // a ≠ default}
  snd.toFun a := if h : a = default then none else some ⟨a, h⟩
  snd.invFun := Option.elim' default (↑)
                       /-
                         α✝ : Sort u_1
                         α₁ : Sort u_2
                         α₂ : Sort u_3
                         β : Sort u_4
                         β₁ : Sort u_5
                         β₂ : Sort u_6
                         γ : Sort u_7
                         δ : Sort u_8
                         α : Type u
                         inst✝¹ : Inhabited α
                         inst✝ : DecidableEq α
                         a : α
                         ⊢ Eq (Option.elim' Inhabited.default Subtype.val ((fun a => dite (Eq a Inhabit …
                       -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  snd.left_inv a := by dsimp only; split_ifs <;> simp [*]
                                                 /-
                                                   🎉 no goals
                                                 -/
  snd.right_inv
                 /-
                   α✝ : Sort u_1
                   α₁ : Sort u_2
                   α₂ : Sort u_3
                   β : Sort u_4
                   β₁ : Sort u_5
                   β₂ : Sort u_6
                   γ : Sort u_7
                   δ : Sort u_8
                   α : Type u
                   inst✝¹ : Inhabited α
                   inst✝ : DecidableEq α
                   ⊢ Eq ((fun a => dite (Eq a Inhabited.default) (fun h => Option.none) fun h =>  …
                 -/
    | none => by simp
                 /-
                   🎉 no goals
                 -/
    | some ⟨_, ha⟩ => dif_neg ha


/-- For any predicate `p` on `α`,
the sum of the two subtypes `{a // p a}` and its complement `{a // ¬ p a}`
is naturally equivalent to `α`.

See `subtypeOrEquiv` for sum types over subtypes `{x // p x}` and `{x // q x}`
that are not necessarily `IsCompl p q`. -/
def sumCompl {α : Type*} (p : α → Prop) [DecidablePred p] :
    { a // p a } ⊕ { a // ¬p a } ≃ α where
  toFun := Sum.elim Subtype.val Subtype.val
  invFun a := if h : p a then Sum.inl ⟨a, h⟩ else Sum.inr ⟨a, h⟩
  left_inv := by
    /-
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α : Type u_9
      p : α → Prop
      inst✝ : DecidablePred p
      ⊢ Function.LeftInverse (fun a => dite (p a) (fun h => Sum.inl ⟨a, h⟩) fun h => …
    -/
    rintro (⟨x, hx⟩ | ⟨x, hx⟩) <;> dsimp
      /-
        case inl.mk
        α✝ : Sort u_1
        α₁ : Sort u_2
        α₂ : Sort u_3
        β : Sort u_4
        β₁ : Sort u_5
        β₂ : Sort u_6
        γ : Sort u_7
        δ : Sort u_8
        α : Type u_9
        p : α → Prop
        inst✝ : DecidablePred p
        x : α
        hx : p x
        ⊢ Eq (dite (p x) (fun h => Sum.inl ⟨x, h⟩) fun h => Sum.inr ⟨x, h⟩) (Sum.inl ⟨ …
      -/
    · rw [dif_pos]
      /-
        🎉 no goals
      -/
      /-
        case inr.mk
        α✝ : Sort u_1
        α₁ : Sort u_2
        α₂ : Sort u_3
        β : Sort u_4
        β₁ : Sort u_5
        β₂ : Sort u_6
        γ : Sort u_7
        δ : Sort u_8
        α : Type u_9
        p : α → Prop
        inst✝ : DecidablePred p
        x : α
        hx : Not (p x)
        ⊢ Eq (dite (p x) (fun h => Sum.inl ⟨x, h⟩) fun h => Sum.inr ⟨x, h⟩) (Sum.inr ⟨ …
      -/
    · rw [dif_neg]
      /-
        🎉 no goals
      -/
  right_inv a := by
    /-
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α : Type u_9
      p : α → Prop
      inst✝ : DecidablePred p
      a : α
      ⊢ Eq (Sum.elim Subtype.val Subtype.val ((fun a => dite (p a) (fun h => Sum.inl …
    -/
    dsimp
    /-
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α : Type u_9
      p : α → Prop
      inst✝ : DecidablePred p
      a : α
      ⊢ Eq (Sum.elim Subtype.val Subtype.val (dite (p a) (fun h => Sum.inl ⟨a, h⟩) f …
    -/
                  /-
                    🎉 no goals
                  -/
    split_ifs <;> rfl
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem sumCompl_apply_inl {α} (p : α → Prop) [DecidablePred p] (x : { a // p a }) :
    sumCompl p (Sum.inl x) = x :=
  rfl


@[simp]
theorem sumCompl_apply_inr {α} (p : α → Prop) [DecidablePred p] (x : { a // ¬p a }) :
    sumCompl p (Sum.inr x) = x :=
  rfl


@[simp]
theorem sumCompl_apply_symm_of_pos {α} (p : α → Prop) [DecidablePred p] (a : α) (h : p a) :
    (sumCompl p).symm a = Sum.inl ⟨a, h⟩ :=
  dif_pos h


@[simp]
theorem sumCompl_apply_symm_of_neg {α} (p : α → Prop) [DecidablePred p] (a : α) (h : ¬p a) :
    (sumCompl p).symm a = Sum.inr ⟨a, h⟩ :=
  dif_neg h


/-- Combines an `Equiv` between two subtypes with an `Equiv` between their complements to form a
  permutation. -/
def subtypeCongr {α} {p q : α → Prop} [DecidablePred p] [DecidablePred q]
    (e : { x // p x } ≃ { x // q x }) (f : { x // ¬p x } ≃ { x // ¬q x }) : Perm α :=
  (sumCompl p).symm.trans ((sumCongr e f).trans (sumCompl q))


/-- Combining permutations on `ε` that permute only inside or outside the subtype
split induced by `p : ε → Prop` constructs a permutation on `ε`. -/
def Perm.subtypeCongr : Equiv.Perm ε :=
  permCongr (sumCompl p) (sumCongr ep en)


theorem Perm.subtypeCongr.apply (a : ε) : ep.subtypeCongr en a =
    if h : p a then (ep ⟨a, h⟩ : ε) else en ⟨a, h⟩ := by
  /-
    ε : Type u_9
    p : ε → Prop
    inst✝ : DecidablePred p
    ep : Equiv.Perm (Subtype fun a => p a)
    en : Equiv.Perm (Subtype fun a => Not (p a))
    a : ε
    ⊢ Eq ((ep.subtypeCongr en) a) (dite (p a) (fun h => ↑(ep ⟨a, h⟩)) fun h => ↑(e …
  -/
                       /-
                         🎉 no goals
                       -/
  by_cases h : p a <;> simp [Perm.subtypeCongr, h]
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem Perm.subtypeCongr.left_apply {a : ε} (h : p a) : ep.subtypeCongr en a = ep ⟨a, h⟩ := by
  /-
    ε : Type u_9
    p : ε → Prop
    inst✝ : DecidablePred p
    ep : Equiv.Perm (Subtype fun a => p a)
    en : Equiv.Perm (Subtype fun a => Not (p a))
    a : ε
    h : p a
    ⊢ Eq ((ep.subtypeCongr en) a) ↑(ep ⟨a, h⟩)
  -/
  simp [Perm.subtypeCongr.apply, h]
  /-
    🎉 no goals
  -/


@[simp]
theorem Perm.subtypeCongr.left_apply_subtype (a : { a // p a }) : ep.subtypeCongr en a = ep a :=
    Perm.subtypeCongr.left_apply ep en a.property


@[simp]
theorem Perm.subtypeCongr.right_apply {a : ε} (h : ¬p a) : ep.subtypeCongr en a = en ⟨a, h⟩ := by
  /-
    ε : Type u_9
    p : ε → Prop
    inst✝ : DecidablePred p
    ep : Equiv.Perm (Subtype fun a => p a)
    en : Equiv.Perm (Subtype fun a => Not (p a))
    a : ε
    h : Not (p a)
    ⊢ Eq ((ep.subtypeCongr en) a) ↑(en ⟨a, h⟩)
  -/
  simp [Perm.subtypeCongr.apply, h]
  /-
    🎉 no goals
  -/


@[simp]
theorem Perm.subtypeCongr.right_apply_subtype (a : { a // ¬p a }) : ep.subtypeCongr en a = en a :=
  Perm.subtypeCongr.right_apply ep en a.property


@[simp]
theorem Perm.subtypeCongr.refl :
    Perm.subtypeCongr (Equiv.refl { a // p a }) (Equiv.refl { a // ¬p a }) = Equiv.refl ε := by
  /-
    ε : Type u_9
    p : ε → Prop
    inst✝ : DecidablePred p
    ⊢ Eq (Equiv.Perm.subtypeCongr (Equiv.refl (Subtype fun a => p a)) (Equiv.refl  …
  -/
  ext x
  /-
    case H
    ε : Type u_9
    p : ε → Prop
    inst✝ : DecidablePred p
    x : ε
    ⊢ Eq ((Equiv.Perm.subtypeCongr (Equiv.refl (Subtype fun a => p a)) (Equiv.refl …
  -/
                       /-
                         🎉 no goals
                       -/
  by_cases h : p x <;> simp [h]
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem Perm.subtypeCongr.symm : (ep.subtypeCongr en).symm = Perm.subtypeCongr ep.symm en.symm := by
  /-
    ε : Type u_9
    p : ε → Prop
    inst✝ : DecidablePred p
    ep : Equiv.Perm (Subtype fun a => p a)
    en : Equiv.Perm (Subtype fun a => Not (p a))
    ⊢ Eq (Equiv.symm (ep.subtypeCongr en)) (Equiv.Perm.subtypeCongr (Equiv.symm ep …
  -/
  ext x
  /-
    case H
    ε : Type u_9
    p : ε → Prop
    inst✝ : DecidablePred p
    ep : Equiv.Perm (Subtype fun a => p a)
    en : Equiv.Perm (Subtype fun a => Not (p a))
    x : ε
    ⊢ Eq ((Equiv.symm (ep.subtypeCongr en)) x) ((Equiv.Perm.subtypeCongr (Equiv.sy …
  -/
  by_cases h : p x
    /-
      case pos
      ε : Type u_9
      p : ε → Prop
      inst✝ : DecidablePred p
      ep : Equiv.Perm (Subtype fun a => p a)
      en : Equiv.Perm (Subtype fun a => Not (p a))
      x : ε
      h : p x
      ⊢ Eq ((Equiv.symm (ep.subtypeCongr en)) x) ((Equiv.Perm.subtypeCongr (Equiv.sy …
    -/
  · have : p (ep.symm ⟨x, h⟩) := Subtype.property _
    /-
      case pos
      ε : Type u_9
      p : ε → Prop
      inst✝ : DecidablePred p
      ep : Equiv.Perm (Subtype fun a => p a)
      en : Equiv.Perm (Subtype fun a => Not (p a))
      x : ε
      h : p x
      this : p ↑((Equiv.symm ep) ⟨x, h⟩)
      ⊢ Eq ((Equiv.symm (ep.subtypeCongr en)) x) ((Equiv.Perm.subtypeCongr (Equiv.sy …
    -/
    simp [Perm.subtypeCongr.apply, h, symm_apply_eq, this]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ε : Type u_9
      p : ε → Prop
      inst✝ : DecidablePred p
      ep : Equiv.Perm (Subtype fun a => p a)
      en : Equiv.Perm (Subtype fun a => Not (p a))
      x : ε
      h : Not (p x)
      ⊢ Eq ((Equiv.symm (ep.subtypeCongr en)) x) ((Equiv.Perm.subtypeCongr (Equiv.sy …
    -/
  · have : ¬p (en.symm ⟨x, h⟩) := Subtype.property (en.symm _)
    /-
      case neg
      ε : Type u_9
      p : ε → Prop
      inst✝ : DecidablePred p
      ep : Equiv.Perm (Subtype fun a => p a)
      en : Equiv.Perm (Subtype fun a => Not (p a))
      x : ε
      h : Not (p x)
      this : Not (p ↑((Equiv.symm en) ⟨x, h⟩))
      ⊢ Eq ((Equiv.symm (ep.subtypeCongr en)) x) ((Equiv.Perm.subtypeCongr (Equiv.sy …
    -/
    simp [Perm.subtypeCongr.apply, h, symm_apply_eq, this]
    /-
      🎉 no goals
    -/


@[simp]
theorem Perm.subtypeCongr.trans :
    (ep.subtypeCongr en).trans (ep'.subtypeCongr en')
    = Perm.subtypeCongr (ep.trans ep') (en.trans en') := by
  /-
    ε : Type u_9
    p : ε → Prop
    inst✝ : DecidablePred p
    ep ep' : Equiv.Perm (Subtype fun a => p a)
    en en' : Equiv.Perm (Subtype fun a => Not (p a))
    ⊢ Eq (Equiv.trans (ep.subtypeCongr en) (ep'.subtypeCongr en')) (Equiv.Perm.sub …
  -/
  ext x
  /-
    case H
    ε : Type u_9
    p : ε → Prop
    inst✝ : DecidablePred p
    ep ep' : Equiv.Perm (Subtype fun a => p a)
    en en' : Equiv.Perm (Subtype fun a => Not (p a))
    x : ε
    ⊢ Eq ((Equiv.trans (ep.subtypeCongr en) (ep'.subtypeCongr en')) x) ((Equiv.Per …
  -/
  by_cases h : p x
    /-
      case pos
      ε : Type u_9
      p : ε → Prop
      inst✝ : DecidablePred p
      ep ep' : Equiv.Perm (Subtype fun a => p a)
      en en' : Equiv.Perm (Subtype fun a => Not (p a))
      x : ε
      h : p x
      ⊢ Eq ((Equiv.trans (ep.subtypeCongr en) (ep'.subtypeCongr en')) x) ((Equiv.Per …
    -/
  · have : p (ep ⟨x, h⟩) := Subtype.property _
    /-
      case pos
      ε : Type u_9
      p : ε → Prop
      inst✝ : DecidablePred p
      ep ep' : Equiv.Perm (Subtype fun a => p a)
      en en' : Equiv.Perm (Subtype fun a => Not (p a))
      x : ε
      h : p x
      this : p ↑(ep ⟨x, h⟩)
      ⊢ Eq ((Equiv.trans (ep.subtypeCongr en) (ep'.subtypeCongr en')) x) ((Equiv.Per …
    -/
    simp [Perm.subtypeCongr.apply, h, this]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ε : Type u_9
      p : ε → Prop
      inst✝ : DecidablePred p
      ep ep' : Equiv.Perm (Subtype fun a => p a)
      en en' : Equiv.Perm (Subtype fun a => Not (p a))
      x : ε
      h : Not (p x)
      ⊢ Eq ((Equiv.trans (ep.subtypeCongr en) (ep'.subtypeCongr en')) x) ((Equiv.Per …
    -/
  · have : ¬p (en ⟨x, h⟩) := Subtype.property (en _)
    /-
      case neg
      ε : Type u_9
      p : ε → Prop
      inst✝ : DecidablePred p
      ep ep' : Equiv.Perm (Subtype fun a => p a)
      en en' : Equiv.Perm (Subtype fun a => Not (p a))
      x : ε
      h : Not (p x)
      this : Not (p ↑(en ⟨x, h⟩))
      ⊢ Eq ((Equiv.trans (ep.subtypeCongr en) (ep'.subtypeCongr en')) x) ((Equiv.Per …
    -/
    simp [Perm.subtypeCongr.apply, h, symm_apply_eq, this]
    /-
      🎉 no goals
    -/


/-- For a fixed function `x₀ : {a // p a} → β` defined on a subtype of `α`,
the subtype of functions `x : α → β` that agree with `x₀` on the subtype `{a // p a}`
is naturally equivalent to the type of functions `{a // ¬ p a} → β`. -/
@[simps]
def subtypePreimage : { x : α → β // x ∘ Subtype.val = x₀ } ≃ ({ a // ¬p a } → β) where
  toFun (x : { x : α → β // x ∘ Subtype.val = x₀ }) a := (x : α → β) a
  invFun x := ⟨fun a => if h : p a then x₀ ⟨a, h⟩ else x ⟨a, h⟩, funext fun ⟨_, h⟩ => dif_pos h⟩
  left_inv := fun ⟨x, hx⟩ =>
    Subtype.val_injective <|
      funext fun a => by
        /-
          α : Sort u_1
          α₁ : Sort u_2
          α₂ : Sort u_3
          β : Sort u_4
          β₁ : Sort u_5
          β₂ : Sort u_6
          γ : Sort u_7
          δ : Sort u_8
          p : α → Prop
          inst✝ : DecidablePred p
          x₀ : (Subtype fun a => p a) → β
          x✝ : Subtype fun x => Eq (Function.comp x Subtype.val) x₀
          x : α → β
          hx : Eq (Function.comp x Subtype.val) x₀
          a : α
          ⊢ Eq (↑((fun x => ⟨fun a => dite (p a) (fun h => x₀ ⟨a, h⟩) fun h => x ⟨a, h⟩, …
        -/
        dsimp only
        /-
          α : Sort u_1
          α₁ : Sort u_2
          α₂ : Sort u_3
          β : Sort u_4
          β₁ : Sort u_5
          β₂ : Sort u_6
          γ : Sort u_7
          δ : Sort u_8
          p : α → Prop
          inst✝ : DecidablePred p
          x₀ : (Subtype fun a => p a) → β
          x✝ : Subtype fun x => Eq (Function.comp x Subtype.val) x₀
          x : α → β
          hx : Eq (Function.comp x Subtype.val) x₀
          a : α
          ⊢ Eq (dite (p a) (fun h => x₀ ⟨a, h⟩) fun h => x a) (x a)
        -/
        split_ifs
          /-
            case pos
            α : Sort u_1
            α₁ : Sort u_2
            α₂ : Sort u_3
            β : Sort u_4
            β₁ : Sort u_5
            β₂ : Sort u_6
            γ : Sort u_7
            δ : Sort u_8
            p : α → Prop
            inst✝ : DecidablePred p
            x₀ : (Subtype fun a => p a) → β
            x✝ : Subtype fun x => Eq (Function.comp x Subtype.val) x₀
            x : α → β
            hx : Eq (Function.comp x Subtype.val) x₀
            a : α
            h✝ : p a
            ⊢ Eq (x₀ ⟨a, h✝⟩) (x a)
          -/
        · rw [← hx]; rfl
                     /-
                       🎉 no goals
                     -/
          /-
            case neg
            α : Sort u_1
            α₁ : Sort u_2
            α₂ : Sort u_3
            β : Sort u_4
            β₁ : Sort u_5
            β₂ : Sort u_6
            γ : Sort u_7
            δ : Sort u_8
            p : α → Prop
            inst✝ : DecidablePred p
            x₀ : (Subtype fun a => p a) → β
            x✝ : Subtype fun x => Eq (Function.comp x Subtype.val) x₀
            x : α → β
            hx : Eq (Function.comp x Subtype.val) x₀
            a : α
            h✝ : Not (p a)
            ⊢ Eq (x a) (x a)
          -/
        · rfl
          /-
            🎉 no goals
          -/
  right_inv x :=
    funext fun ⟨a, h⟩ =>
      show dite (p a) _ _ = _ by
        /-
          α : Sort u_1
          α₁ : Sort u_2
          α₂ : Sort u_3
          β : Sort u_4
          β₁ : Sort u_5
          β₂ : Sort u_6
          γ : Sort u_7
          δ : Sort u_8
          p : α → Prop
          inst✝ : DecidablePred p
          x₀ : (Subtype fun a => p a) → β
          x : (Subtype fun a => Not (p a)) → β
          x✝ : Subtype fun a => Not (p a)
          a : α
          h : Not (p a)
          ⊢ Eq (dite (p a) (fun h_1 => x₀ ⟨↑⟨a, h⟩, h_1⟩) fun h_1 => x ⟨↑⟨a, h⟩, h_1⟩) ( …
        -/
        dsimp only
        /-
          α : Sort u_1
          α₁ : Sort u_2
          α₂ : Sort u_3
          β : Sort u_4
          β₁ : Sort u_5
          β₂ : Sort u_6
          γ : Sort u_7
          δ : Sort u_8
          p : α → Prop
          inst✝ : DecidablePred p
          x₀ : (Subtype fun a => p a) → β
          x : (Subtype fun a => Not (p a)) → β
          x✝ : Subtype fun a => Not (p a)
          a : α
          h : Not (p a)
          ⊢ Eq (dite (p a) (fun h => x₀ ⟨a, h⟩) fun h => x ⟨a, h⟩) (x ⟨a, h⟩)
        -/
        rw [dif_neg h]
        /-
          🎉 no goals
        -/


theorem subtypePreimage_symm_apply_coe_pos (x : { a // ¬p a } → β) (a : α) (h : p a) :
    ((subtypePreimage p x₀).symm x : α → β) a = x₀ ⟨a, h⟩ :=
  dif_pos h


theorem subtypePreimage_symm_apply_coe_neg (x : { a // ¬p a } → β) (a : α) (h : ¬p a) :
    ((subtypePreimage p x₀).symm x : α → β) a = x ⟨a, h⟩ :=
  dif_neg h


/-- A family of equivalences `∀ a, β₁ a ≃ β₂ a` generates an equivalence between `∀ a, β₁ a` and
`∀ a, β₂ a`. -/
@[simps]
def piCongrRight {β₁ β₂ : α → Sort*} (F : ∀ a, β₁ a ≃ β₂ a) : (∀ a, β₁ a) ≃ (∀ a, β₂ a) :=
                                                                        /-
                                                                          α : Sort u_1
                                                                          α₁ : Sort u_2
                                                                          α₂ : Sort u_3
                                                                          β : Sort u_4
                                                                          β₁✝ : Sort u_5
                                                                          β₂✝ : Sort u_6
                                                                          γ : Sort u_7
                                                                          δ : Sort u_8
                                                                          β₁ : α → Sort u_9
                                                                          β₂ : α → Sort u_10
                                                                          F : (a : α) → Equiv (β₁ a) (β₂ a)
                                                                          H : (a : α) → β₁ a
                                                                          ⊢ ∀ (x : α), Eq (Pi.map (fun a => ⇑(F a).symm) (Pi.map (fun a => ⇑(F a)) H) x) …
                                                                        -/
  ⟨Pi.map fun a ↦ F a, Pi.map fun a ↦ (F a).symm, fun H => funext <| by simp,
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
                          /-
                            α : Sort u_1
                            α₁ : Sort u_2
                            α₂ : Sort u_3
                            β : Sort u_4
                            β₁✝ : Sort u_5
                            β₂✝ : Sort u_6
                            γ : Sort u_7
                            δ : Sort u_8
                            β₁ : α → Sort u_9
                            β₂ : α → Sort u_10
                            F : (a : α) → Equiv (β₁ a) (β₂ a)
                            H : (a : α) → β₂ a
                            ⊢ ∀ (x : α), Eq (Pi.map (fun a => ⇑(F a)) (Pi.map (fun a => ⇑(F a).symm) H) x) …
                          -/
    fun H => funext <| by simp⟩
                          /-
                            🎉 no goals
                          -/


/-- Given `φ : α → β → Sort*`, we have an equivalence between `∀ a b, φ a b` and `∀ b a, φ a b`.
This is `Function.swap` as an `Equiv`. -/
@[simps apply]
def piComm (φ : α → β → Sort*) : (∀ a b, φ a b) ≃ ∀ b a, φ a b :=
  ⟨swap, swap, fun _ => rfl, fun _ => rfl⟩


@[simp]
theorem piComm_symm {φ : α → β → Sort*} : (piComm φ).symm = (piComm <| swap φ) :=
  rfl


/-- Dependent `curry` equivalence: the type of dependent functions on `Σ i, β i` is equivalent
to the type of dependent functions of two arguments (i.e., functions to the space of functions).

This is `Sigma.curry` and `Sigma.uncurry` together as an equiv. -/
def piCurry {α} {β : α → Type*} (γ : ∀ a, β a → Type*) :
    (∀ x : Σ i, β i, γ x.1 x.2) ≃ ∀ a b, γ a b where
  toFun := Sigma.curry
  invFun := Sigma.uncurry
  left_inv := Sigma.uncurry_curry
  right_inv := Sigma.curry_uncurry

-- `simps` overapplies these but `simps (config := .asFn)` under-applies them

@[simp] theorem piCurry_apply {α} {β : α → Type*} (γ : ∀ a, β a → Type*)
    (f : ∀ x : Σ i, β i, γ x.1 x.2) :
    piCurry γ f = Sigma.curry f :=
  rfl


@[simp] theorem piCurry_symm_apply {α} {β : α → Type*} (γ : ∀ a, β a → Type*) (f : ∀ a b, γ a b) :
    (piCurry γ).symm f = Sigma.uncurry f :=
  rfl


/-- A family of equivalences `∀ (a : α₁), β₁ ≃ β₂` generates an equivalence
between `β₁ × α₁` and `β₂ × α₁`. -/
def prodCongrLeft : β₁ × α₁ ≃ β₂ × α₁ where
  toFun ab := ⟨e ab.2 ab.1, ab.2⟩
  invFun ab := ⟨(e ab.2).symm ab.1, ab.2⟩
  left_inv := by
    /-
      α : Sort u_1
      α₁✝ : Sort u_2
      α₂✝ : Sort u_3
      β : Sort u_4
      β₁✝ : Sort u_5
      β₂✝ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α₁ : Type u_9
      α₂ : Type u_10
      β₁ : Type u_11
      β₂ : Type u_12
      e : α₁ → Equiv β₁ β₂
      ⊢ Function.LeftInverse (fun ab => { fst := (e ab.2).symm ab.1, snd := ab.2 })  …
    -/
    rintro ⟨a, b⟩
    /-
      case mk
      α : Sort u_1
      α₁✝ : Sort u_2
      α₂✝ : Sort u_3
      β : Sort u_4
      β₁✝ : Sort u_5
      β₂✝ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α₁ : Type u_9
      α₂ : Type u_10
      β₁ : Type u_11
      β₂ : Type u_12
      e : α₁ → Equiv β₁ β₂
      a : β₁
      b : α₁
      ⊢ Eq ((fun ab => { fst := (e ab.2).symm ab.1, snd := ab.2 }) ((fun ab => { fst …
    -/
    simp
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      α : Sort u_1
      α₁✝ : Sort u_2
      α₂✝ : Sort u_3
      β : Sort u_4
      β₁✝ : Sort u_5
      β₂✝ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α₁ : Type u_9
      α₂ : Type u_10
      β₁ : Type u_11
      β₂ : Type u_12
      e : α₁ → Equiv β₁ β₂
      ⊢ Function.RightInverse (fun ab => { fst := (e ab.2).symm ab.1, snd := ab.2 }) …
    -/
    rintro ⟨a, b⟩
    /-
      case mk
      α : Sort u_1
      α₁✝ : Sort u_2
      α₂✝ : Sort u_3
      β : Sort u_4
      β₁✝ : Sort u_5
      β₂✝ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α₁ : Type u_9
      α₂ : Type u_10
      β₁ : Type u_11
      β₂ : Type u_12
      e : α₁ → Equiv β₁ β₂
      a : β₂
      b : α₁
      ⊢ Eq ((fun ab => { fst := (e ab.2) ab.1, snd := ab.2 }) ((fun ab => { fst := ( …
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
theorem prodCongrLeft_apply (b : β₁) (a : α₁) : prodCongrLeft e (b, a) = (e a b, a) :=
  rfl


theorem prodCongr_refl_right (e : β₁ ≃ β₂) :
    prodCongr e (Equiv.refl α₁) = prodCongrLeft fun _ => e := by
  /-
    α₁ : Type u_9
    β₁ : Type u_11
    β₂ : Type u_12
    e : Equiv β₁ β₂
    ⊢ Eq (e.prodCongr (Equiv.refl α₁)) (Equiv.prodCongrLeft fun x => e)
  -/
  ext ⟨a, b⟩ : 1
  /-
    case H.mk
    α₁ : Type u_9
    β₁ : Type u_11
    β₂ : Type u_12
    e : Equiv β₁ β₂
    a : β₁
    b : α₁
    ⊢ Eq ((e.prodCongr (Equiv.refl α₁)) { fst := a, snd := b }) ((Equiv.prodCongrL …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A family of equivalences `∀ (a : α₁), β₁ ≃ β₂` generates an equivalence
between `α₁ × β₁` and `α₁ × β₂`. -/
def prodCongrRight : α₁ × β₁ ≃ α₁ × β₂ where
  toFun ab := ⟨ab.1, e ab.1 ab.2⟩
  invFun ab := ⟨ab.1, (e ab.1).symm ab.2⟩
  left_inv := by
    /-
      α : Sort u_1
      α₁✝ : Sort u_2
      α₂✝ : Sort u_3
      β : Sort u_4
      β₁✝ : Sort u_5
      β₂✝ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α₁ : Type u_9
      α₂ : Type u_10
      β₁ : Type u_11
      β₂ : Type u_12
      e : α₁ → Equiv β₁ β₂
      ⊢ Function.LeftInverse (fun ab => { fst := ab.1, snd := (e ab.1).symm ab.2 })  …
    -/
    rintro ⟨a, b⟩
    /-
      case mk
      α : Sort u_1
      α₁✝ : Sort u_2
      α₂✝ : Sort u_3
      β : Sort u_4
      β₁✝ : Sort u_5
      β₂✝ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α₁ : Type u_9
      α₂ : Type u_10
      β₁ : Type u_11
      β₂ : Type u_12
      e : α₁ → Equiv β₁ β₂
      a : α₁
      b : β₁
      ⊢ Eq ((fun ab => { fst := ab.1, snd := (e ab.1).symm ab.2 }) ((fun ab => { fst …
    -/
    simp
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      α : Sort u_1
      α₁✝ : Sort u_2
      α₂✝ : Sort u_3
      β : Sort u_4
      β₁✝ : Sort u_5
      β₂✝ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α₁ : Type u_9
      α₂ : Type u_10
      β₁ : Type u_11
      β₂ : Type u_12
      e : α₁ → Equiv β₁ β₂
      ⊢ Function.RightInverse (fun ab => { fst := ab.1, snd := (e ab.1).symm ab.2 }) …
    -/
    rintro ⟨a, b⟩
    /-
      case mk
      α : Sort u_1
      α₁✝ : Sort u_2
      α₂✝ : Sort u_3
      β : Sort u_4
      β₁✝ : Sort u_5
      β₂✝ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α₁ : Type u_9
      α₂ : Type u_10
      β₁ : Type u_11
      β₂ : Type u_12
      e : α₁ → Equiv β₁ β₂
      a : α₁
      b : β₂
      ⊢ Eq ((fun ab => { fst := ab.1, snd := (e ab.1) ab.2 }) ((fun ab => { fst := a …
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
theorem prodCongrRight_apply (a : α₁) (b : β₁) : prodCongrRight e (a, b) = (a, e a b) :=
  rfl


theorem prodCongr_refl_left (e : β₁ ≃ β₂) :
    prodCongr (Equiv.refl α₁) e = prodCongrRight fun _ => e := by
  /-
    α₁ : Type u_9
    β₁ : Type u_11
    β₂ : Type u_12
    e : Equiv β₁ β₂
    ⊢ Eq ((Equiv.refl α₁).prodCongr e) (Equiv.prodCongrRight fun x => e)
  -/
  ext ⟨a, b⟩ : 1
  /-
    case H.mk
    α₁ : Type u_9
    β₁ : Type u_11
    β₂ : Type u_12
    e : Equiv β₁ β₂
    a : α₁
    b : β₁
    ⊢ Eq (((Equiv.refl α₁).prodCongr e) { fst := a, snd := b }) ((Equiv.prodCongrR …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem prodCongrLeft_trans_prodComm :
    (prodCongrLeft e).trans (prodComm _ _) = (prodComm _ _).trans (prodCongrRight e) := by
  /-
    α₁ : Type u_9
    β₁ : Type u_11
    β₂ : Type u_12
    e : α₁ → Equiv β₁ β₂
    ⊢ Eq ((Equiv.prodCongrLeft e).trans (Equiv.prodComm β₂ α₁)) ((Equiv.prodComm β …
  -/
  ext ⟨a, b⟩ : 1
  /-
    case H.mk
    α₁ : Type u_9
    β₁ : Type u_11
    β₂ : Type u_12
    e : α₁ → Equiv β₁ β₂
    a : β₁
    b : α₁
    ⊢ Eq (((Equiv.prodCongrLeft e).trans (Equiv.prodComm β₂ α₁)) { fst := a, snd : …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem prodCongrRight_trans_prodComm :
    (prodCongrRight e).trans (prodComm _ _) = (prodComm _ _).trans (prodCongrLeft e) := by
  /-
    α₁ : Type u_9
    β₁ : Type u_11
    β₂ : Type u_12
    e : α₁ → Equiv β₁ β₂
    ⊢ Eq ((Equiv.prodCongrRight e).trans (Equiv.prodComm α₁ β₂)) ((Equiv.prodComm  …
  -/
  ext ⟨a, b⟩ : 1
  /-
    case H.mk
    α₁ : Type u_9
    β₁ : Type u_11
    β₂ : Type u_12
    e : α₁ → Equiv β₁ β₂
    a : α₁
    b : β₁
    ⊢ Eq (((Equiv.prodCongrRight e).trans (Equiv.prodComm α₁ β₂)) { fst := a, snd  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem sigmaCongrRight_sigmaEquivProd :
    (sigmaCongrRight e).trans (sigmaEquivProd α₁ β₂)
    = (sigmaEquivProd α₁ β₁).trans (prodCongrRight e) := by
  /-
    α₁ : Type u_9
    β₁ : Type u_11
    β₂ : Type u_12
    e : α₁ → Equiv β₁ β₂
    ⊢ Eq ((Equiv.sigmaCongrRight e).trans (Equiv.sigmaEquivProd α₁ β₂)) ((Equiv.si …
  -/
  ext ⟨a, b⟩ : 1
  /-
    case H.mk
    α₁ : Type u_9
    β₁ : Type u_11
    β₂ : Type u_12
    e : α₁ → Equiv β₁ β₂
    a : α₁
    b : β₁
    ⊢ Eq (((Equiv.sigmaCongrRight e).trans (Equiv.sigmaEquivProd α₁ β₂)) ⟨a, b⟩) ( …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem sigmaEquivProd_sigmaCongrRight :
    (sigmaEquivProd α₁ β₁).symm.trans (sigmaCongrRight e)
    = (prodCongrRight e).trans (sigmaEquivProd α₁ β₂).symm := by
  /-
    α₁ : Type u_9
    β₁ : Type u_11
    β₂ : Type u_12
    e : α₁ → Equiv β₁ β₂
    ⊢ Eq ((Equiv.sigmaEquivProd α₁ β₁).symm.trans (Equiv.sigmaCongrRight e)) ((Equ …
  -/
  ext ⟨a, b⟩ : 1
  /-
    case H.mk
    α₁ : Type u_9
    β₁ : Type u_11
    β₂ : Type u_12
    e : α₁ → Equiv β₁ β₂
    a : α₁
    b : β₁
    ⊢ Eq (((Equiv.sigmaEquivProd α₁ β₁).symm.trans (Equiv.sigmaCongrRight e)) { fs …
  -/
  simp only [trans_apply, sigmaCongrRight_apply, prodCongrRight_apply]
  /-
    case H.mk
    α₁ : Type u_9
    β₁ : Type u_11
    β₂ : Type u_12
    e : α₁ → Equiv β₁ β₂
    a : α₁
    b : β₁
    ⊢ Eq ⟨((Equiv.sigmaEquivProd α₁ β₁).symm { fst := a, snd := b }).fst, (e ((Equ …
  -/
  rfl
  /-
    🎉 no goals
  -/

-- See also `Equiv.ofPreimageEquiv`.

/-- A family of equivalences between fibers gives an equivalence between domains. -/
@[simps!]
def ofFiberEquiv {α β γ} {f : α → γ} {g : β → γ}
    (e : ∀ c, { a // f a = c } ≃ { b // g b = c }) : α ≃ β :=
  (sigmaFiberEquiv f).symm.trans <| (Equiv.sigmaCongrRight e).trans (sigmaFiberEquiv g)


theorem ofFiberEquiv_map {α β γ} {f : α → γ} {g : β → γ}
    (e : ∀ c, { a // f a = c } ≃ { b // g b = c }) (a : α) : g (ofFiberEquiv e a) = f a :=
  (_ : { b // g b = _ }).property


/-- A variation on `Equiv.prodCongr` where the equivalence in the second component can depend
  on the first component. A typical example is a shear mapping, explaining the name of this
  declaration. -/
@[simps (config := .asFn)]
def prodShear (e₁ : α₁ ≃ α₂) (e₂ : α₁ → β₁ ≃ β₂) : α₁ × β₁ ≃ α₂ × β₂ where
  toFun := fun x : α₁ × β₁ => (e₁ x.1, e₂ x.1 x.2)
  invFun := fun y : α₂ × β₂ => (e₁.symm y.1, (e₂ <| e₁.symm y.1).symm y.2)
  left_inv := by
    /-
      α : Sort u_1
      α₁✝ : Sort u_2
      α₂✝ : Sort u_3
      β : Sort u_4
      β₁✝ : Sort u_5
      β₂✝ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α₁ : Type u_9
      α₂ : Type u_10
      β₁ : Type u_11
      β₂ : Type u_12
      e : α₁ → Equiv β₁ β₂
      e₁ : Equiv α₁ α₂
      e₂ : α₁ → Equiv β₁ β₂
      ⊢ Function.LeftInverse (fun y => { fst := e₁.symm y.1, snd := (e₂ (e₁.symm y.1 …
    -/
    rintro ⟨x₁, y₁⟩
    /-
      case mk
      α : Sort u_1
      α₁✝ : Sort u_2
      α₂✝ : Sort u_3
      β : Sort u_4
      β₁✝ : Sort u_5
      β₂✝ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α₁ : Type u_9
      α₂ : Type u_10
      β₁ : Type u_11
      β₂ : Type u_12
      e : α₁ → Equiv β₁ β₂
      e₁ : Equiv α₁ α₂
      e₂ : α₁ → Equiv β₁ β₂
      x₁ : α₁
      y₁ : β₁
      ⊢ Eq ((fun y => { fst := e₁.symm y.1, snd := (e₂ (e₁.symm y.1)).symm y.2 }) (( …
    -/
    simp only [symm_apply_apply]
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      α : Sort u_1
      α₁✝ : Sort u_2
      α₂✝ : Sort u_3
      β : Sort u_4
      β₁✝ : Sort u_5
      β₂✝ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α₁ : Type u_9
      α₂ : Type u_10
      β₁ : Type u_11
      β₂ : Type u_12
      e : α₁ → Equiv β₁ β₂
      e₁ : Equiv α₁ α₂
      e₂ : α₁ → Equiv β₁ β₂
      ⊢ Function.RightInverse (fun y => { fst := e₁.symm y.1, snd := (e₂ (e₁.symm y. …
    -/
    rintro ⟨x₁, y₁⟩
    /-
      case mk
      α : Sort u_1
      α₁✝ : Sort u_2
      α₂✝ : Sort u_3
      β : Sort u_4
      β₁✝ : Sort u_5
      β₂✝ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α₁ : Type u_9
      α₂ : Type u_10
      β₁ : Type u_11
      β₂ : Type u_12
      e : α₁ → Equiv β₁ β₂
      e₁ : Equiv α₁ α₂
      e₂ : α₁ → Equiv β₁ β₂
      x₁ : α₂
      y₁ : β₂
      ⊢ Eq ((fun x => { fst := e₁ x.1, snd := (e₂ x.1) x.2 }) ((fun y => { fst := e₁ …
    -/
    simp only [apply_symm_apply]
    /-
      🎉 no goals
    -/


/-- `prodExtendRight a e` extends `e : Perm β` to `Perm (α × β)` by sending `(a, b)` to
`(a, e b)` and keeping the other `(a', b)` fixed. -/
def prodExtendRight : Perm (α₁ × β₁) where
  toFun ab := if ab.fst = a then (a, e ab.snd) else ab
  invFun ab := if ab.fst = a then (a, e.symm ab.snd) else ab
  left_inv := by
    /-
      α : Sort u_1
      α₁✝ : Sort u_2
      α₂ : Sort u_3
      β : Sort u_4
      β₁✝ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α₁ : Type u_9
      β₁ : Type u_10
      inst✝ : DecidableEq α₁
      a : α₁
      e : Equiv.Perm β₁
      ⊢ Function.LeftInverse (fun ab => ite (Eq ab.1 a) { fst := a, snd := (Equiv.sy …
    -/
    rintro ⟨k', x⟩
    /-
      case mk
      α : Sort u_1
      α₁✝ : Sort u_2
      α₂ : Sort u_3
      β : Sort u_4
      β₁✝ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α₁ : Type u_9
      β₁ : Type u_10
      inst✝ : DecidableEq α₁
      a : α₁
      e : Equiv.Perm β₁
      k' : α₁
      x : β₁
      ⊢ Eq ((fun ab => ite (Eq ab.1 a) { fst := a, snd := (Equiv.symm e) ab.2 } ab)  …
    -/
    dsimp only
    /-
      case mk
      α : Sort u_1
      α₁✝ : Sort u_2
      α₂ : Sort u_3
      β : Sort u_4
      β₁✝ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α₁ : Type u_9
      β₁ : Type u_10
      inst✝ : DecidableEq α₁
      a : α₁
      e : Equiv.Perm β₁
      k' : α₁
      x : β₁
      ⊢ Eq (ite (Eq (ite (Eq k' a) { fst := a, snd := e x } { fst := k', snd := x }) …
    -/
    split_ifs with h₁ h₂
      /-
        case pos
        α : Sort u_1
        α₁✝ : Sort u_2
        α₂ : Sort u_3
        β : Sort u_4
        β₁✝ : Sort u_5
        β₂ : Sort u_6
        γ : Sort u_7
        δ : Sort u_8
        α₁ : Type u_9
        β₁ : Type u_10
        inst✝ : DecidableEq α₁
        a : α₁
        e : Equiv.Perm β₁
        k' : α₁
        x : β₁
        h₁ : Eq k' a
        h₂ : Eq { fst := a, snd := e x }.1 a
        ⊢ Eq { fst := a, snd := (Equiv.symm e) { fst := a, snd := e x }.2 } { fst := k …
      -/
    · simp [h₁]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Sort u_1
        α₁✝ : Sort u_2
        α₂ : Sort u_3
        β : Sort u_4
        β₁✝ : Sort u_5
        β₂ : Sort u_6
        γ : Sort u_7
        δ : Sort u_8
        α₁ : Type u_9
        β₁ : Type u_10
        inst✝ : DecidableEq α₁
        a : α₁
        e : Equiv.Perm β₁
        k' : α₁
        x : β₁
        h₁ : Eq k' a
        h₂ : Not (Eq { fst := a, snd := e x }.1 a)
        ⊢ Eq { fst := a, snd := e x } { fst := k', snd := x }
      -/
    · simp at h₂
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Sort u_1
        α₁✝ : Sort u_2
        α₂ : Sort u_3
        β : Sort u_4
        β₁✝ : Sort u_5
        β₂ : Sort u_6
        γ : Sort u_7
        δ : Sort u_8
        α₁ : Type u_9
        β₁ : Type u_10
        inst✝ : DecidableEq α₁
        a : α₁
        e : Equiv.Perm β₁
        k' : α₁
        x : β₁
        h₁ : Not (Eq k' a)
        ⊢ Eq { fst := k', snd := x } { fst := k', snd := x }
      -/
    · simp
      /-
        🎉 no goals
      -/
  right_inv := by
    /-
      α : Sort u_1
      α₁✝ : Sort u_2
      α₂ : Sort u_3
      β : Sort u_4
      β₁✝ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α₁ : Type u_9
      β₁ : Type u_10
      inst✝ : DecidableEq α₁
      a : α₁
      e : Equiv.Perm β₁
      ⊢ Function.RightInverse (fun ab => ite (Eq ab.1 a) { fst := a, snd := (Equiv.s …
    -/
    rintro ⟨k', x⟩
    /-
      case mk
      α : Sort u_1
      α₁✝ : Sort u_2
      α₂ : Sort u_3
      β : Sort u_4
      β₁✝ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α₁ : Type u_9
      β₁ : Type u_10
      inst✝ : DecidableEq α₁
      a : α₁
      e : Equiv.Perm β₁
      k' : α₁
      x : β₁
      ⊢ Eq ((fun ab => ite (Eq ab.1 a) { fst := a, snd := e ab.2 } ab) ((fun ab => i …
    -/
    dsimp only
    /-
      case mk
      α : Sort u_1
      α₁✝ : Sort u_2
      α₂ : Sort u_3
      β : Sort u_4
      β₁✝ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α₁ : Type u_9
      β₁ : Type u_10
      inst✝ : DecidableEq α₁
      a : α₁
      e : Equiv.Perm β₁
      k' : α₁
      x : β₁
      ⊢ Eq (ite (Eq (ite (Eq k' a) { fst := a, snd := (Equiv.symm e) x } { fst := k' …
    -/
    split_ifs with h₁ h₂
      /-
        case pos
        α : Sort u_1
        α₁✝ : Sort u_2
        α₂ : Sort u_3
        β : Sort u_4
        β₁✝ : Sort u_5
        β₂ : Sort u_6
        γ : Sort u_7
        δ : Sort u_8
        α₁ : Type u_9
        β₁ : Type u_10
        inst✝ : DecidableEq α₁
        a : α₁
        e : Equiv.Perm β₁
        k' : α₁
        x : β₁
        h₁ : Eq k' a
        h₂ : Eq { fst := a, snd := (Equiv.symm e) x }.1 a
        ⊢ Eq { fst := a, snd := e { fst := a, snd := (Equiv.symm e) x }.2 } { fst := k …
      -/
    · simp [h₁]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Sort u_1
        α₁✝ : Sort u_2
        α₂ : Sort u_3
        β : Sort u_4
        β₁✝ : Sort u_5
        β₂ : Sort u_6
        γ : Sort u_7
        δ : Sort u_8
        α₁ : Type u_9
        β₁ : Type u_10
        inst✝ : DecidableEq α₁
        a : α₁
        e : Equiv.Perm β₁
        k' : α₁
        x : β₁
        h₁ : Eq k' a
        h₂ : Not (Eq { fst := a, snd := (Equiv.symm e) x }.1 a)
        ⊢ Eq { fst := a, snd := (Equiv.symm e) x } { fst := k', snd := x }
      -/
    · simp at h₂
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Sort u_1
        α₁✝ : Sort u_2
        α₂ : Sort u_3
        β : Sort u_4
        β₁✝ : Sort u_5
        β₂ : Sort u_6
        γ : Sort u_7
        δ : Sort u_8
        α₁ : Type u_9
        β₁ : Type u_10
        inst✝ : DecidableEq α₁
        a : α₁
        e : Equiv.Perm β₁
        k' : α₁
        x : β₁
        h₁ : Not (Eq k' a)
        ⊢ Eq { fst := k', snd := x } { fst := k', snd := x }
      -/
    · simp
      /-
        🎉 no goals
      -/


@[simp]
theorem prodExtendRight_apply_eq (b : β₁) : prodExtendRight a e (a, b) = (a, e b) :=
  if_pos rfl


theorem prodExtendRight_apply_ne {a a' : α₁} (h : a' ≠ a) (b : β₁) :
    prodExtendRight a e (a', b) = (a', b) :=
  if_neg h


theorem eq_of_prodExtendRight_ne {e : Perm β₁} {a a' : α₁} {b : β₁}
    (h : prodExtendRight a e (a', b) ≠ (a', b)) : a' = a := by
  /-
    α₁ : Type u_9
    β₁ : Type u_10
    inst✝ : DecidableEq α₁
    e : Equiv.Perm β₁
    a a' : α₁
    b : β₁
    h : Ne ((Equiv.Perm.prodExtendRight a e) { fst := a', snd := b }) { fst := a', …
    ⊢ Eq a' a
  -/
  contrapose! h
  /-
    α₁ : Type u_9
    β₁ : Type u_10
    inst✝ : DecidableEq α₁
    e : Equiv.Perm β₁
    a a' : α₁
    b : β₁
    h : Ne a' a
    ⊢ Eq ((Equiv.Perm.prodExtendRight a e) { fst := a', snd := b }) { fst := a', s …
  -/
  exact prodExtendRight_apply_ne _ h _
  /-
    🎉 no goals
  -/


@[simp]
theorem fst_prodExtendRight (ab : α₁ × β₁) : (prodExtendRight a e ab).fst = ab.fst := by
  /-
    α₁ : Type u_9
    β₁ : Type u_10
    inst✝ : DecidableEq α₁
    a : α₁
    e : Equiv.Perm β₁
    ab : Prod α₁ β₁
    ⊢ Eq ((Equiv.Perm.prodExtendRight a e) ab).1 ab.1
  -/
  rw [prodExtendRight]
  /-
    α₁ : Type u_9
    β₁ : Type u_10
    inst✝ : DecidableEq α₁
    a : α₁
    e : Equiv.Perm β₁
    ab : Prod α₁ β₁
    ⊢ Eq ({ toFun := fun ab => ite (Eq ab.1 a) { fst := a, snd := e ab.2 } ab, inv …
  -/
  dsimp
  /-
    α₁ : Type u_9
    β₁ : Type u_10
    inst✝ : DecidableEq α₁
    a : α₁
    e : Equiv.Perm β₁
    ab : Prod α₁ β₁
    ⊢ Eq (ite (Eq ab.1 a) { fst := a, snd := e ab.2 } ab).1 ab.1
  -/
  split_ifs with h
    /-
      case pos
      α₁ : Type u_9
      β₁ : Type u_10
      inst✝ : DecidableEq α₁
      a : α₁
      e : Equiv.Perm β₁
      ab : Prod α₁ β₁
      h : Eq ab.1 a
      ⊢ Eq { fst := a, snd := e ab.2 }.1 ab.1
    -/
  · rw [h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α₁ : Type u_9
      β₁ : Type u_10
      inst✝ : DecidableEq α₁
      a : α₁
      e : Equiv.Perm β₁
      ab : Prod α₁ β₁
      h : Not (Eq ab.1 a)
      ⊢ Eq ab.1 ab.1
    -/
  · rfl
    /-
      🎉 no goals
    -/


/-- The type of functions to a product `α × β` is equivalent to the type of pairs of functions
`γ → α` and `γ → β`. -/
def arrowProdEquivProdArrow (α β γ : Type*) : (γ → α × β) ≃ (γ → α) × (γ → β) where
  toFun := fun f => (fun c => (f c).1, fun c => (f c).2)
  invFun := fun p c => (p.1 c, p.2 c)
  left_inv := fun _ => rfl
                           /-
                             α✝ : Sort u_1
                             α₁ : Sort u_2
                             α₂ : Sort u_3
                             β✝ : Sort u_4
                             β₁ : Sort u_5
                             β₂ : Sort u_6
                             γ✝ : Sort u_7
                             δ : Sort u_8
                             α : Type u_9
                             β : Type u_10
                             γ : Type u_11
                             p : Prod (γ → α) (γ → β)
                             ⊢ Eq ((fun f => { fst := fun c => (f c).1, snd := fun c => (f c).2 }) ((fun p  …
                           -/
  right_inv := fun p => by cases p; rfl
                                    /-
                                      🎉 no goals
                                    -/


/-- The type of dependent functions on a sum type `ι ⊕ ι'` is equivalent to the type of pairs of
functions on `ι` and on `ι'`. This is a dependent version of `Equiv.sumArrowEquivProdArrow`. -/
@[simps]
def sumPiEquivProdPi {ι ι'} (π : ι ⊕ ι' → Type*) :
    (∀ i, π i) ≃ (∀ i, π (inl i)) × ∀ i', π (inr i') where
  toFun f := ⟨fun i => f (inl i), fun i' => f (inr i')⟩
  invFun g := Sum.rec g.1 g.2
                   /-
                     α : Sort u_1
                     α₁ : Sort u_2
                     α₂ : Sort u_3
                     β : Sort u_4
                     β₁ : Sort u_5
                     β₂ : Sort u_6
                     γ : Sort u_7
                     δ : Sort u_8
                     ι : Type ?u.59577
                     ι' : Type ?u.59576
                     π : Sum ι ι' → Type u_9
                     f : (i : Sum ι ι') → π i
                     ⊢ Eq ((fun g t => Sum.rec g.1 g.2 t) ((fun f => { fst := fun i => f (Sum.inl i …
                   -/
                                   /-
                                     🎉 no goals
                                   -/
  left_inv f := by ext (i | i) <;> rfl
                                   /-
                                     🎉 no goals
                                   -/
  right_inv _ := Prod.ext rfl rfl


/-- The equivalence between a product of two dependent functions types and a single dependent
function type. Basically a symmetric version of `Equiv.sumPiEquivProdPi`. -/
@[simps!]
def prodPiEquivSumPi {ι ι'} (π : ι → Type u) (π' : ι' → Type u) :
    ((∀ i, π i) × ∀ i', π' i') ≃ ∀ i, Sum.elim π π' i :=
  sumPiEquivProdPi (Sum.elim π π') |>.symm


/-- The type of functions on a sum type `α ⊕ β` is equivalent to the type of pairs of functions
on `α` and on `β`. -/
def sumArrowEquivProdArrow (α β γ : Type*) : (α ⊕ β → γ) ≃ (α → γ) × (β → γ) :=
                                                                       /-
                                                                         α✝ : Sort u_1
                                                                         α₁ : Sort u_2
                                                                         α₂ : Sort u_3
                                                                         β✝ : Sort u_4
                                                                         β₁ : Sort u_5
                                                                         β₂ : Sort u_6
                                                                         γ✝ : Sort u_7
                                                                         δ : Sort u_8
                                                                         α : Type u_9
                                                                         β : Type u_10
                                                                         γ : Type u_11
                                                                         f : Sum α β → γ
                                                                         ⊢ Eq ((fun p => Sum.elim p.1 p.2) ((fun f => { fst := Function.comp f Sum.inl, …
                                                                       -/
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
  ⟨fun f => (f ∘ inl, f ∘ inr), fun p => Sum.elim p.1 p.2, fun f => by ext ⟨⟩ <;> rfl, fun p => by
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
    /-
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ✝ : Sort u_7
      δ : Sort u_8
      α : Type u_9
      β : Type u_10
      γ : Type u_11
      p : Prod (α → γ) (β → γ)
      ⊢ Eq ((fun f => { fst := Function.comp f Sum.inl, snd := Function.comp f Sum.i …
    -/
    cases p
    /-
      case mk
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ✝ : Sort u_7
      δ : Sort u_8
      α : Type u_9
      β : Type u_10
      γ : Type u_11
      fst✝ : α → γ
      snd✝ : β → γ
      ⊢ Eq ((fun f => { fst := Function.comp f Sum.inl, snd := Function.comp f Sum.i …
    -/
    rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem sumArrowEquivProdArrow_apply_fst {α β γ} (f : α ⊕ β → γ) (a : α) :
    (sumArrowEquivProdArrow α β γ f).1 a = f (inl a) :=
  rfl


@[simp]
theorem sumArrowEquivProdArrow_apply_snd {α β γ} (f : α ⊕ β → γ) (b : β) :
    (sumArrowEquivProdArrow α β γ f).2 b = f (inr b) :=
  rfl


@[simp]
theorem sumArrowEquivProdArrow_symm_apply_inl {α β γ} (f : α → γ) (g : β → γ) (a : α) :
    ((sumArrowEquivProdArrow α β γ).symm (f, g)) (inl a) = f a :=
  rfl


@[simp]
theorem sumArrowEquivProdArrow_symm_apply_inr {α β γ} (f : α → γ) (g : β → γ) (b : β) :
    ((sumArrowEquivProdArrow α β γ).symm (f, g)) (inr b) = g b :=
  rfl


/-- Type product is right distributive with respect to type sum up to an equivalence. -/
def sumProdDistrib (α β γ) : (α ⊕ β) × γ ≃ α × γ ⊕ β × γ :=
  ⟨fun p => p.1.map (fun x => (x, p.2)) fun x => (x, p.2),
    fun s => s.elim (Prod.map inl id) (Prod.map inr id), by
      /-
        α✝ : Sort u_1
        α₁ : Sort u_2
        α₂ : Sort u_3
        β✝ : Sort u_4
        β₁ : Sort u_5
        β₂ : Sort u_6
        γ✝ : Sort u_7
        δ : Sort u_8
        α : Type ?u.61818
        β : Type ?u.61817
        γ : Type ?u.61815
        ⊢ Function.LeftInverse (fun s => Sum.elim (Prod.map Sum.inl id) (Prod.map Sum. …
      -/
                            /-
                              🎉 no goals
                            -/
                            /-
                              🎉 no goals
                            -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
      rintro ⟨_ | _, _⟩ <;> rfl, by rintro (⟨_, _⟩ | ⟨_, _⟩) <;> rfl⟩
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem sumProdDistrib_apply_left {α β γ} (a : α) (c : γ) :
    sumProdDistrib α β γ (Sum.inl a, c) = Sum.inl (a, c) :=
  rfl


@[simp]
theorem sumProdDistrib_apply_right {α β γ} (b : β) (c : γ) :
    sumProdDistrib α β γ (Sum.inr b, c) = Sum.inr (b, c) :=
  rfl


@[simp]
theorem sumProdDistrib_symm_apply_left {α β γ} (a : α × γ) :
    (sumProdDistrib α β γ).symm (inl a) = (inl a.1, a.2) :=
  rfl


@[simp]
theorem sumProdDistrib_symm_apply_right {α β γ} (b : β × γ) :
    (sumProdDistrib α β γ).symm (inr b) = (inr b.1, b.2) :=
  rfl


/-- Type product is left distributive with respect to type sum up to an equivalence. -/
def prodSumDistrib (α β γ) : α × (β ⊕ γ) ≃ (α × β) ⊕ (α × γ) :=
  calc
    α × (β ⊕ γ) ≃ (β ⊕ γ) × α := prodComm _ _
    _ ≃ (β × α) ⊕ (γ × α) := sumProdDistrib _ _ _
    _ ≃ (α × β) ⊕ (α × γ) := sumCongr (prodComm _ _) (prodComm _ _)


@[simp]
theorem prodSumDistrib_apply_left {α β γ} (a : α) (b : β) :
    prodSumDistrib α β γ (a, Sum.inl b) = Sum.inl (a, b) :=
  rfl


@[simp]
theorem prodSumDistrib_apply_right {α β γ} (a : α) (c : γ) :
    prodSumDistrib α β γ (a, Sum.inr c) = Sum.inr (a, c) :=
  rfl


@[simp]
theorem prodSumDistrib_symm_apply_left {α β γ} (a : α × β) :
    (prodSumDistrib α β γ).symm (inl a) = (a.1, inl a.2) :=
  rfl


@[simp]
theorem prodSumDistrib_symm_apply_right {α β γ} (a : α × γ) :
    (prodSumDistrib α β γ).symm (inr a) = (a.1, inr a.2) :=
  rfl


/-- An indexed sum of disjoint sums of types is equivalent to the sum of the indexed sums. -/
@[simps]
def sigmaSumDistrib {ι} (α β : ι → Type*) :
    (Σ i, α i ⊕ β i) ≃ (Σ i, α i) ⊕ (Σ i, β i) :=
  ⟨fun p => p.2.map (Sigma.mk p.1) (Sigma.mk p.1),
    Sum.elim (Sigma.map id fun _ => Sum.inl) (Sigma.map id fun _ => Sum.inr), fun p => by
    /-
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      ι : Type ?u.64429
      α : ι → Type u_9
      β : ι → Type u_10
      p : Sigma fun i => Sum (α i) (β i)
      ⊢ Eq (Sum.elim (Sigma.map id fun x => Sum.inl) (Sigma.map id fun x => Sum.inr) …
    -/
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
    rcases p with ⟨i, a | b⟩ <;> rfl, fun p => by rcases p with (⟨i, a⟩ | ⟨i, b⟩) <;> rfl⟩
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


/-- The product of an indexed sum of types (formally, a `Sigma`-type `Σ i, α i`) by a type `β` is
equivalent to the sum of products `Σ i, (α i × β)`. -/
@[simps apply symm_apply]
def sigmaProdDistrib {ι} (α : ι → Type*) (β) : (Σ i, α i) × β ≃ Σ i, α i × β :=
  ⟨fun p => ⟨p.1.1, (p.1.2, p.2)⟩, fun p => (⟨p.1, p.2.1⟩, p.2.2), fun p => by
    /-
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      ι : Type ?u.64970
      α : ι → Type u_9
      β : Type ?u.64967
      p : Prod (Sigma fun i => α i) β
      ⊢ Eq ((fun p => { fst := ⟨p.fst, p.snd.1⟩, snd := p.snd.2 }) ((fun p => ⟨p.1.f …
    -/
    rcases p with ⟨⟨_, _⟩, _⟩
    /-
      case mk.mk
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      ι : Type ?u.64970
      α : ι → Type u_9
      β : Type ?u.64967
      snd✝¹ : β
      fst✝ : ι
      snd✝ : α fst✝
      ⊢ Eq ((fun p => { fst := ⟨p.fst, p.snd.1⟩, snd := p.snd.2 }) ((fun p => ⟨p.1.f …
    -/
    rfl, fun p => by
    /-
      🎉 no goals
    -/
    /-
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      ι : Type ?u.64970
      α : ι → Type u_9
      β : Type ?u.64967
      p : Sigma fun i => Prod (α i) β
      ⊢ Eq ((fun p => ⟨p.1.fst, { fst := p.1.snd, snd := p.2 }⟩) ((fun p => { fst := …
    -/
    rcases p with ⟨_, ⟨_, _⟩⟩
    /-
      case mk.mk
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      ι : Type ?u.64970
      α : ι → Type u_9
      β : Type ?u.64967
      fst✝¹ : ι
      fst✝ : α fst✝¹
      snd✝ : β
      ⊢ Eq ((fun p => ⟨p.1.fst, { fst := p.1.snd, snd := p.2 }⟩) ((fun p => { fst := …
    -/
    rfl⟩
    /-
      🎉 no goals
    -/


/-- An equivalence that separates out the 0th fiber of `(Σ (n : ℕ), f n)`. -/
def sigmaNatSucc (f : ℕ → Type u) : (Σ n, f n) ≃ f 0 ⊕ Σ n, f (n + 1) :=
  ⟨fun x =>
    @Sigma.casesOn ℕ f (fun _ => f 0 ⊕ Σ n, f (n + 1)) x fun n =>
      @Nat.casesOn (fun i => f i → f 0 ⊕ Σ n : ℕ, f (n + 1)) n (fun x : f 0 => Sum.inl x)
        fun (n : ℕ) (x : f n.succ) => Sum.inr ⟨n, x⟩,
                                                               /-
                                                                 α : Sort u_1
                                                                 α₁ : Sort u_2
                                                                 α₂ : Sort u_3
                                                                 β : Sort u_4
                                                                 β₁ : Sort u_5
                                                                 β₂ : Sort u_6
                                                                 γ : Sort u_7
                                                                 δ : Sort u_8
                                                                 f : Nat → Type u
                                                                 ⊢ Function.LeftInverse (Sum.elim (Sigma.mk 0) (Sigma.map Nat.succ fun x => id) …
                                                               -/
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
    Sum.elim (Sigma.mk 0) (Sigma.map Nat.succ fun _ => id), by rintro ⟨n | n, x⟩ <;> rfl, by
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
    /-
      α : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      f : Nat → Type u
      ⊢ Function.RightInverse (Sum.elim (Sigma.mk 0) (Sigma.map Nat.succ fun x => id …
    -/
                            /-
                              🎉 no goals
                            -/
    rintro (x | ⟨n, x⟩) <;> rfl⟩
                            /-
                              🎉 no goals
                            -/


/-- The product `Bool × α` is equivalent to `α ⊕ α`. -/
@[simps]
def boolProdEquivSum (α) : Bool × α ≃ α ⊕ α where
  toFun p := p.1.casesOn (inl p.2) (inr p.2)
  invFun := Sum.elim (Prod.mk false) (Prod.mk true)
                 /-
                   α✝ : Sort u_1
                   α₁ : Sort u_2
                   α₂ : Sort u_3
                   β : Sort u_4
                   β₁ : Sort u_5
                   β₂ : Sort u_6
                   γ : Sort u_7
                   δ : Sort u_8
                   α : Type ?u.65946
                   ⊢ Function.LeftInverse (Sum.elim (Prod.mk Bool.false) (Prod.mk Bool.true)) fun …
                 -/
                                       /-
                                         🎉 no goals
                                       -/
  left_inv := by rintro ⟨_ | _, _⟩ <;> rfl
                                       /-
                                         🎉 no goals
                                       -/
                  /-
                    α✝ : Sort u_1
                    α₁ : Sort u_2
                    α₂ : Sort u_3
                    β : Sort u_4
                    β₁ : Sort u_5
                    β₂ : Sort u_6
                    γ : Sort u_7
                    δ : Sort u_8
                    α : Type ?u.65946
                    ⊢ Function.RightInverse (Sum.elim (Prod.mk Bool.false) (Prod.mk Bool.true)) fu …
                  -/
                                     /-
                                       🎉 no goals
                                     -/
  right_inv := by rintro (_ | _) <;> rfl
                                     /-
                                       🎉 no goals
                                     -/


/-- The function type `Bool → α` is equivalent to `α × α`. -/
@[simps]
def boolArrowEquivProd (α) : (Bool → α) ≃ α × α where
  toFun f := (f false, f true)
  invFun p b := b.casesOn p.1 p.2
  left_inv _ := funext <| Bool.forall_bool.2 ⟨rfl, rfl⟩
  right_inv := fun _ => rfl


/-- The set of natural numbers is equivalent to `ℕ ⊕ PUnit`. -/
def natEquivNatSumPUnit : ℕ ≃ ℕ ⊕ PUnit where
  toFun n := Nat.casesOn n (inr PUnit.unit) inl
  invFun := Sum.elim Nat.succ fun _ => 0
                   /-
                     α : Sort u_1
                     α₁ : Sort u_2
                     α₂ : Sort u_3
                     β : Sort u_4
                     β₁ : Sort u_5
                     β₂ : Sort u_6
                     γ : Sort u_7
                     δ : Sort u_8
                     n : Nat
                     ⊢ Eq (Sum.elim Nat.succ (fun x => 0) ((fun n => Nat.casesOn n (Sum.inr PUnit.u …
                   -/
                               /-
                                 🎉 no goals
                               -/
  left_inv n := by cases n <;> rfl
                               /-
                                 🎉 no goals
                               -/
                  /-
                    α : Sort u_1
                    α₁ : Sort u_2
                    α₂ : Sort u_3
                    β : Sort u_4
                    β₁ : Sort u_5
                    β₂ : Sort u_6
                    γ : Sort u_7
                    δ : Sort u_8
                    ⊢ Function.RightInverse (Sum.elim Nat.succ fun x => 0) fun n => Nat.casesOn n  …
                  -/
                                     /-
                                       🎉 no goals
                                     -/
  right_inv := by rintro (_ | _) <;> rfl
                                     /-
                                       🎉 no goals
                                     -/


/-- `ℕ ⊕ PUnit` is equivalent to `ℕ`. -/
def natSumPUnitEquivNat : ℕ ⊕ PUnit ≃ ℕ :=
  natEquivNatSumPUnit.symm


/-- The type of integer numbers is equivalent to `ℕ ⊕ ℕ`. -/
def intEquivNatSumNat : ℤ ≃ ℕ ⊕ ℕ where
  toFun z := Int.casesOn z inl inr
  invFun := Sum.elim Int.ofNat Int.negSucc
                 /-
                   α : Sort u_1
                   α₁ : Sort u_2
                   α₂ : Sort u_3
                   β : Sort u_4
                   β₁ : Sort u_5
                   β₂ : Sort u_6
                   γ : Sort u_7
                   δ : Sort u_8
                   ⊢ Function.LeftInverse (Sum.elim Int.ofNat Int.negSucc) fun z => Int.casesOn z …
                 -/
                                    /-
                                      🎉 no goals
                                    -/
  left_inv := by rintro (m | n) <;> rfl
                                    /-
                                      🎉 no goals
                                    -/
                  /-
                    α : Sort u_1
                    α₁ : Sort u_2
                    α₂ : Sort u_3
                    β : Sort u_4
                    β₁ : Sort u_5
                    β₂ : Sort u_6
                    γ : Sort u_7
                    δ : Sort u_8
                    ⊢ Function.RightInverse (Sum.elim Int.ofNat Int.negSucc) fun z => Int.casesOn  …
                  -/
                                     /-
                                       🎉 no goals
                                     -/
  right_inv := by rintro (m | n) <;> rfl
                                     /-
                                       🎉 no goals
                                     -/


/-- An equivalence between `α` and `β` generates an equivalence between `List α` and `List β`. -/
def listEquivOfEquiv {α β} (e : α ≃ β) : List α ≃ List β where
  toFun := List.map e
  invFun := List.map e.symm
                   /-
                     α✝ : Sort u_1
                     α₁ : Sort u_2
                     α₂ : Sort u_3
                     β✝ : Sort u_4
                     β₁ : Sort u_5
                     β₂ : Sort u_6
                     γ : Sort u_7
                     δ : Sort u_8
                     α : Type ?u.66986
                     β : Type ?u.66987
                     e : Equiv α β
                     l : List α
                     ⊢ Eq (List.map (⇑e.symm) (List.map (⇑e) l)) l
                   -/
  left_inv l := by rw [List.map_map, e.symm_comp_self, List.map_id]
                   /-
                     🎉 no goals
                   -/
                    /-
                      α✝ : Sort u_1
                      α₁ : Sort u_2
                      α₂ : Sort u_3
                      β✝ : Sort u_4
                      β₁ : Sort u_5
                      β₂ : Sort u_6
                      γ : Sort u_7
                      δ : Sort u_8
                      α : Type ?u.66986
                      β : Type ?u.66987
                      e : Equiv α β
                      l : List β
                      ⊢ Eq (List.map (⇑e) (List.map (⇑e.symm) l)) l
                    -/
  right_inv l := by rw [List.map_map, e.self_comp_symm, List.map_id]
                    /-
                      🎉 no goals
                    -/


/-- If `α` is equivalent to `β`, then `Unique α` is equivalent to `Unique β`. -/
def uniqueCongr (e : α ≃ β) : Unique α ≃ Unique β where
  toFun h := @Equiv.unique _ _ h e.symm
  invFun h := @Equiv.unique _ _ h e
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


/-- If `α` is equivalent to `β`, then `IsEmpty α` is equivalent to `IsEmpty β`. -/
theorem isEmpty_congr (e : α ≃ β) : IsEmpty α ↔ IsEmpty β :=
  ⟨fun h => @Function.isEmpty _ _ h e.symm, fun h => @Function.isEmpty _ _ h e⟩


protected theorem isEmpty (e : α ≃ β) [IsEmpty β] : IsEmpty α :=
  e.isEmpty_congr.mpr ‹_›


/-- If `α` is equivalent to `β` and the predicates `p : α → Prop` and `q : β → Prop` are equivalent
at corresponding points, then `{a // p a}` is equivalent to `{b // q b}`.
For the statement where `α = β`, that is, `e : perm α`, see `Perm.subtypePerm`. -/
def subtypeEquiv {p : α → Prop} {q : β → Prop} (e : α ≃ β) (h : ∀ a, p a ↔ q (e a)) :
    { a : α // p a } ≃ { b : β // q b } where
  toFun a := ⟨e a, (h _).mp a.property⟩
  invFun b := ⟨e.symm b, (h _).mpr ((e.apply_symm_apply b).symm ▸ b.property)⟩
                                  /-
                                    α : Sort u_1
                                    α₁ : Sort u_2
                                    α₂ : Sort u_3
                                    β : Sort u_4
                                    β₁ : Sort u_5
                                    β₂ : Sort u_6
                                    γ : Sort u_7
                                    δ : Sort u_8
                                    p : α → Prop
                                    q : β → Prop
                                    e : Equiv α β
                                    h : ∀ (a : α), Iff (p a) (q (e a))
                                    a : Subtype fun a => p a
                                    ⊢ Eq ↑((fun b => ⟨e.symm ↑b, ⋯⟩) ((fun a => ⟨e ↑a, ⋯⟩) a)) ↑a
                                  -/
  left_inv a := Subtype.ext <| by simp
                                  /-
                                    🎉 no goals
                                  -/
                                   /-
                                     α : Sort u_1
                                     α₁ : Sort u_2
                                     α₂ : Sort u_3
                                     β : Sort u_4
                                     β₁ : Sort u_5
                                     β₂ : Sort u_6
                                     γ : Sort u_7
                                     δ : Sort u_8
                                     p : α → Prop
                                     q : β → Prop
                                     e : Equiv α β
                                     h : ∀ (a : α), Iff (p a) (q (e a))
                                     b : Subtype fun b => q b
                                     ⊢ Eq ↑((fun a => ⟨e ↑a, ⋯⟩) ((fun b => ⟨e.symm ↑b, ⋯⟩) b)) ↑b
                                   -/
  right_inv b := Subtype.ext <| by simp
                                   /-
                                     🎉 no goals
                                   -/


lemma coe_subtypeEquiv_eq_map {X Y} {p : X → Prop} {q : Y → Prop} (e : X ≃ Y)
    (h : ∀ x, p x ↔ q (e x)) : ⇑(e.subtypeEquiv h) = Subtype.map e (h · |>.mp) :=
  rfl


@[simp]
theorem subtypeEquiv_refl {p : α → Prop} (h : ∀ a, p a ↔ p (Equiv.refl _ a) := fun _ => Iff.rfl) :
    (Equiv.refl α).subtypeEquiv h = Equiv.refl { a : α // p a } := by
  /-
    α : Sort u_1
    p : α → Prop
    h : optParam (∀ (a : α), Iff (p a) (p ((Equiv.refl α) a))) ⋯
    ⊢ Eq ((Equiv.refl α).subtypeEquiv h) (Equiv.refl (Subtype fun a => p a))
  -/
  ext
  /-
    case H.a
    α : Sort u_1
    p : α → Prop
    h : optParam (∀ (a : α), Iff (p a) (p ((Equiv.refl α) a))) ⋯
    x✝ : Subtype fun a => p a
    ⊢ Eq ↑(((Equiv.refl α).subtypeEquiv h) x✝) ↑((Equiv.refl (Subtype fun a => p a …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem subtypeEquiv_symm {p : α → Prop} {q : β → Prop} (e : α ≃ β) (h : ∀ a : α, p a ↔ q (e a)) :
    (e.subtypeEquiv h).symm =
      e.symm.subtypeEquiv fun a => by
        /-
          α : Sort u_1
          α₁ : Sort u_2
          α₂ : Sort u_3
          β : Sort u_4
          β₁ : Sort u_5
          β₂ : Sort u_6
          γ : Sort u_7
          δ : Sort u_8
          p : α → Prop
          q : β → Prop
          e : Equiv α β
          h : ∀ (a : α), Iff (p a) (q (e a))
          a : β
          ⊢ Iff (q a) (p (e.symm a))
        -/
        convert (h <| e.symm a).symm
        /-
          case h.e'_1.h.e'_1
          α : Sort u_1
          α₁ : Sort u_2
          α₂ : Sort u_3
          β : Sort u_4
          β₁ : Sort u_5
          β₂ : Sort u_6
          γ : Sort u_7
          δ : Sort u_8
          p : α → Prop
          q : β → Prop
          e : Equiv α β
          h : ∀ (a : α), Iff (p a) (q (e a))
          a : β
          ⊢ Eq a (e (e.symm a))
        -/
        exact (e.apply_symm_apply a).symm :=
        /-
          🎉 no goals
        -/
  rfl


@[simp]
theorem subtypeEquiv_trans {p : α → Prop} {q : β → Prop} {r : γ → Prop} (e : α ≃ β) (f : β ≃ γ)
    (h : ∀ a : α, p a ↔ q (e a)) (h' : ∀ b : β, q b ↔ r (f b)) :
    (e.subtypeEquiv h).trans (f.subtypeEquiv h')
    = (e.trans f).subtypeEquiv fun a => (h a).trans (h' <| e a) :=
  rfl


@[simp]
theorem subtypeEquiv_apply {p : α → Prop} {q : β → Prop}
    (e : α ≃ β) (h : ∀ a : α, p a ↔ q (e a)) (x : { x // p x }) :
    e.subtypeEquiv h x = ⟨e x, (h _).1 x.2⟩ :=
  rfl


/-- If two predicates `p` and `q` are pointwise equivalent, then `{x // p x}` is equivalent to
`{x // q x}`. -/
@[simps!]
def subtypeEquivRight {p q : α → Prop} (e : ∀ x, p x ↔ q x) : { x // p x } ≃ { x // q x } :=
  subtypeEquiv (Equiv.refl _) e


lemma subtypeEquivRight_apply {p q : α → Prop} (e : ∀ x, p x ↔ q x)
    (z : { x // p x }) : subtypeEquivRight e z = ⟨z, (e z.1).mp z.2⟩ := rfl


lemma subtypeEquivRight_symm_apply {p q : α → Prop} (e : ∀ x, p x ↔ q x)
    (z : { x // q x }) : (subtypeEquivRight e).symm z = ⟨z, (e z.1).mpr z.2⟩ := rfl


/-- If `α ≃ β`, then for any predicate `p : β → Prop` the subtype `{a // p (e a)}` is equivalent
to the subtype `{b // p b}`. -/
def subtypeEquivOfSubtype {p : β → Prop} (e : α ≃ β) : { a : α // p (e a) } ≃ { b : β // p b } :=
                       /-
                         α : Sort u_1
                         α₁ : Sort u_2
                         α₂ : Sort u_3
                         β : Sort u_4
                         β₁ : Sort u_5
                         β₂ : Sort u_6
                         γ : Sort u_7
                         δ : Sort u_8
                         p : β → Prop
                         e : Equiv α β
                         ⊢ ∀ (a : α), Iff (p (e a)) (p (e a))
                       -/
  subtypeEquiv e <| by simp
                       /-
                         🎉 no goals
                       -/


/-- If `α ≃ β`, then for any predicate `p : α → Prop` the subtype `{a // p a}` is equivalent
to the subtype `{b // p (e.symm b)}`. This version is used by `equiv_rw`. -/
def subtypeEquivOfSubtype' {p : α → Prop} (e : α ≃ β) :
    { a : α // p a } ≃ { b : β // p (e.symm b) } :=
  e.symm.subtypeEquivOfSubtype.symm


/-- If two predicates are equal, then the corresponding subtypes are equivalent. -/
def subtypeEquivProp {p q : α → Prop} (h : p = q) : Subtype p ≃ Subtype q :=
  subtypeEquiv (Equiv.refl α) fun _ => h ▸ Iff.rfl


/-- A subtype of a subtype is equivalent to the subtype of elements satisfying both predicates. This
version allows the “inner” predicate to depend on `h : p a`. -/
@[simps]
def subtypeSubtypeEquivSubtypeExists (p : α → Prop) (q : Subtype p → Prop) :
    Subtype q ≃ { a : α // ∃ h : p a, q ⟨a, h⟩ } :=
  ⟨fun a =>
    ⟨a.1, a.1.2, by
      /-
        α : Sort u_1
        α₁ : Sort u_2
        α₂ : Sort u_3
        β : Sort u_4
        β₁ : Sort u_5
        β₂ : Sort u_6
        γ : Sort u_7
        δ : Sort u_8
        p : α → Prop
        q : Subtype p → Prop
        a : Subtype q
        ⊢ q ⟨↑↑a, ⋯⟩
      -/
      rcases a with ⟨⟨a, hap⟩, haq⟩
      /-
        case mk.mk
        α : Sort u_1
        α₁ : Sort u_2
        α₂ : Sort u_3
        β : Sort u_4
        β₁ : Sort u_5
        β₂ : Sort u_6
        γ : Sort u_7
        δ : Sort u_8
        p : α → Prop
        q : Subtype p → Prop
        a : α
        hap : p a
        haq : q ⟨a, hap⟩
        ⊢ q ⟨↑↑⟨⟨a, hap⟩, haq⟩, ⋯⟩
      -/
      exact haq⟩,
      /-
        🎉 no goals
      -/
    fun a => ⟨⟨a, a.2.fst⟩, a.2.snd⟩, fun ⟨⟨_, _⟩, _⟩ => rfl, fun ⟨_, _, _⟩ => rfl⟩


/-- A subtype of a subtype is equivalent to the subtype of elements satisfying both predicates. -/
@[simps!]
def subtypeSubtypeEquivSubtypeInter {α : Type u} (p q : α → Prop) :
    { x : Subtype p // q x.1 } ≃ Subtype fun x => p x ∧ q x :=
  (subtypeSubtypeEquivSubtypeExists p _).trans <|
    subtypeEquivRight fun x => @exists_prop (q x) (p x)


/-- If the outer subtype has more restrictive predicate than the inner one,
then we can drop the latter. -/
@[simps!]
def subtypeSubtypeEquivSubtype {α} {p q : α → Prop} (h : ∀ {x}, q x → p x) :
    { x : Subtype p // q x.1 } ≃ Subtype q :=
  (subtypeSubtypeEquivSubtypeInter p _).trans <| subtypeEquivRight fun _ => and_iff_right_of_imp h


/-- If a proposition holds for all elements, then the subtype is
equivalent to the original type. -/
@[simps apply symm_apply]
def subtypeUnivEquiv {α} {p : α → Prop} (h : ∀ x, p x) : Subtype p ≃ α :=
  ⟨fun x => x, fun x => ⟨x, h x⟩, fun _ => Subtype.eq rfl, fun _ => rfl⟩


/-- A subtype of a sigma-type is a sigma-type over a subtype. -/
def subtypeSigmaEquiv {α} (p : α → Type v) (q : α → Prop) : { y : Sigma p // q y.1 } ≃ Σ x :
    Subtype q, p x.1 :=
  ⟨fun x => ⟨⟨x.1.1, x.2⟩, x.1.2⟩, fun x => ⟨⟨x.1.1, x.2⟩, x.1.2⟩, fun _ => rfl,
    fun _ => rfl⟩


/-- A sigma type over a subtype is equivalent to the sigma set over the original type,
if the fiber is empty outside of the subset -/
def sigmaSubtypeEquivOfSubset {α} (p : α → Type v) (q : α → Prop) (h : ∀ x, p x → q x) :
    (Σ x : Subtype q, p x) ≃ Σ x : α, p x :=
  (subtypeSigmaEquiv p q).symm.trans <| subtypeUnivEquiv fun x => h x.1 x.2


/-- If a predicate `p : β → Prop` is true on the range of a map `f : α → β`, then
`Σ y : {y // p y}, {x // f x = y}` is equivalent to `α`. -/
def sigmaSubtypeFiberEquiv {α β : Type*} (f : α → β) (p : β → Prop) (h : ∀ x, p (f x)) :
    (Σ y : Subtype p, { x : α // f x = y }) ≃ α :=
  calc
    _ ≃ Σy : β, { x : α // f x = y } := sigmaSubtypeEquivOfSubset _ p fun _ ⟨x, h'⟩ => h' ▸ h x
    _ ≃ α := sigmaFiberEquiv f


/-- If for each `x` we have `p x ↔ q (f x)`, then `Σ y : {y // q y}, f ⁻¹' {y}` is equivalent
to `{x // p x}`. -/
def sigmaSubtypeFiberEquivSubtype {α β : Type*} (f : α → β) {p : α → Prop} {q : β → Prop}
    (h : ∀ x, p x ↔ q (f x)) : (Σ y : Subtype q, { x : α // f x = y }) ≃ Subtype p :=
  calc
    (Σy : Subtype q, { x : α // f x = y }) ≃ Σy :
        Subtype q, { x : Subtype p // Subtype.mk (f x) ((h x).1 x.2) = y } := by {
          apply sigmaCongrRight
          intro y
          apply Equiv.symm
          refine (subtypeSubtypeEquivSubtypeExists _ _).trans (subtypeEquivRight ?_)
          intro x
          exact ⟨fun ⟨hp, h'⟩ => congr_arg Subtype.val h', fun h' => ⟨(h x).2 (h'.symm ▸ y.2),
            Subtype.eq h'⟩⟩ }
    _ ≃ Subtype p := sigmaFiberEquiv fun x : Subtype p => (⟨f x, (h x).1 x.property⟩ : Subtype q)


/-- A sigma type over an `Option` is equivalent to the sigma set over the original type,
if the fiber is empty at none. -/
def sigmaOptionEquivOfSome {α} (p : Option α → Type v) (h : p none → False) :
    (Σ x : Option α, p x) ≃ Σ x : α, p (some x) :=
  haveI h' : ∀ x, p x → x.isSome := by
    /-
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α : Type ?u.77061
      p : Option α → Type v
      h : p Option.none → False
      ⊢ ∀ (x : Option α), p x → Eq x.isSome Bool.true
    -/
    intro x
    /-
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α : Type ?u.77061
      p : Option α → Type v
      h : p Option.none → False
      x : Option α
      ⊢ p x → Eq x.isSome Bool.true
    -/
    cases x
      /-
        case none
        α✝ : Sort u_1
        α₁ : Sort u_2
        α₂ : Sort u_3
        β : Sort u_4
        β₁ : Sort u_5
        β₂ : Sort u_6
        γ : Sort u_7
        δ : Sort u_8
        α : Type ?u.77061
        p : Option α → Type v
        h : p Option.none → False
        ⊢ p Option.none → Eq Option.none.isSome Bool.true
      -/
    · intro n
      /-
        case none
        α✝ : Sort u_1
        α₁ : Sort u_2
        α₂ : Sort u_3
        β : Sort u_4
        β₁ : Sort u_5
        β₂ : Sort u_6
        γ : Sort u_7
        δ : Sort u_8
        α : Type ?u.77061
        p : Option α → Type v
        h : p Option.none → False
        n : p Option.none
        ⊢ Eq Option.none.isSome Bool.true
      -/
      exfalso
      /-
        case none
        α✝ : Sort u_1
        α₁ : Sort u_2
        α₂ : Sort u_3
        β : Sort u_4
        β₁ : Sort u_5
        β₂ : Sort u_6
        γ : Sort u_7
        δ : Sort u_8
        α : Type ?u.77061
        p : Option α → Type v
        h : p Option.none → False
        n : p Option.none
        ⊢ False
      -/
      exact h n
      /-
        🎉 no goals
      -/
      /-
        case some
        α✝ : Sort u_1
        α₁ : Sort u_2
        α₂ : Sort u_3
        β : Sort u_4
        β₁ : Sort u_5
        β₂ : Sort u_6
        γ : Sort u_7
        δ : Sort u_8
        α : Type ?u.77061
        p : Option α → Type v
        h : p Option.none → False
        val✝ : α
        ⊢ p (Option.some val✝) → Eq (Option.some val✝).isSome Bool.true
      -/
    · intro _
      /-
        case some
        α✝ : Sort u_1
        α₁ : Sort u_2
        α₂ : Sort u_3
        β : Sort u_4
        β₁ : Sort u_5
        β₂ : Sort u_6
        γ : Sort u_7
        δ : Sort u_8
        α : Type ?u.77061
        p : Option α → Type v
        h : p Option.none → False
        val✝ : α
        a✝ : p (Option.some val✝)
        ⊢ Eq (Option.some val✝).isSome Bool.true
      -/
      exact rfl
      /-
        🎉 no goals
      -/
  (sigmaSubtypeEquivOfSubset _ _ h').symm.trans (sigmaCongrLeft' (optionIsSomeEquiv α))


/-- The `Pi`-type `∀ i, π i` is equivalent to the type of sections `f : ι → Σ i, π i` of the
`Sigma` type such that for all `i` we have `(f i).fst = i`. -/
def piEquivSubtypeSigma (ι) (π : ι → Type*) :
    (∀ i, π i) ≃ { f : ι → Σ i, π i // ∀ i, (f i).1 = i } where
  toFun := fun f => ⟨fun i => ⟨i, f i⟩, fun _ => rfl⟩
                          /-
                            α : Sort u_1
                            α₁ : Sort u_2
                            α₂ : Sort u_3
                            β : Sort u_4
                            β₁ : Sort u_5
                            β₂ : Sort u_6
                            γ : Sort u_7
                            δ : Sort u_8
                            ι : Type ?u.77321
                            π : ι → Type u_9
                            f : Subtype fun f => ∀ (i : ι), Eq (f i).fst i
                            i : ι
                            ⊢ π i
                          -/
  invFun := fun f i => by rw [← f.2 i]; exact (f.1 i).2
                                        /-
                                          🎉 no goals
                                        -/
  left_inv := fun _ => funext fun _ => rfl
  right_inv := fun ⟨f, hf⟩ =>
    Subtype.eq <| funext fun i =>
                                                                  /-
                                                                    α : Sort u_1
                                                                    α₁ : Sort u_2
                                                                    α₂ : Sort u_3
                                                                    β : Sort u_4
                                                                    β₁ : Sort u_5
                                                                    β₂ : Sort u_6
                                                                    γ : Sort u_7
                                                                    δ : Sort u_8
                                                                    ι : Type ?u.77321
                                                                    π : ι → Type u_9
                                                                    x✝ : Subtype fun f => ∀ (i : ι), Eq (f i).fst i
                                                                    f : ι → Sigma fun i => π i
                                                                    hf : ∀ (i : ι), Eq (f i).fst i
                                                                    i : ι
                                                                    ⊢ HEq (↑((fun f => ⟨fun i => ⟨i, f i⟩, ⋯⟩) ((fun f i => ⋯.mpr (↑f i).snd) ⟨f,  …
                                                                  -/
      Sigma.eq (hf i).symm <| eq_of_heq <| rec_heq_of_heq _ <| by simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- The type of functions `f : ∀ a, β a` such that for all `a` we have `p a (f a)` is equivalent
to the type of functions `∀ a, {b : β a // p a b}`. -/
def subtypePiEquivPi {β : α → Sort v} {p : ∀ a, β a → Prop} :
    { f : ∀ a, β a // ∀ a, p a (f a) } ≃ ∀ a, { b : β a // p a b } where
  toFun := fun f a => ⟨f.1 a, f.2 a⟩
  invFun := fun f => ⟨fun a => (f a).1, fun a => (f a).2⟩
  left_inv := by
    /-
      α : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      β : α → Sort v
      p : (a : α) → β a → Prop
      ⊢ Function.LeftInverse (fun f => ⟨fun a => ↑(f a), ⋯⟩) fun f a => ⟨↑f a, ⋯⟩
    -/
    rintro ⟨f, h⟩
    /-
      case mk
      α : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      β : α → Sort v
      p : (a : α) → β a → Prop
      f : (a : α) → β a
      h : ∀ (a : α), p a (f a)
      ⊢ Eq ((fun f => ⟨fun a => ↑(f a), ⋯⟩) ((fun f a => ⟨↑f a, ⋯⟩) ⟨f, h⟩)) ⟨f, h⟩
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      α : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      β : α → Sort v
      p : (a : α) → β a → Prop
      ⊢ Function.RightInverse (fun f => ⟨fun a => ↑(f a), ⋯⟩) fun f a => ⟨↑f a, ⋯⟩
    -/
    rintro f
    /-
      α : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      β : α → Sort v
      p : (a : α) → β a → Prop
      f : (a : α) → Subtype fun b => p a b
      ⊢ Eq ((fun f a => ⟨↑f a, ⋯⟩) ((fun f => ⟨fun a => ↑(f a), ⋯⟩) f)) f
    -/
    funext a
    /-
      case h
      α : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      β : α → Sort v
      p : (a : α) → β a → Prop
      f : (a : α) → Subtype fun b => p a b
      a : α
      ⊢ Eq ((fun f a => ⟨↑f a, ⋯⟩) ((fun f => ⟨fun a => ↑(f a), ⋯⟩) f) a) (f a)
    -/
    exact Subtype.ext_val rfl
    /-
      🎉 no goals
    -/


/-- A subtype of a product defined by componentwise conditions
is equivalent to a product of subtypes. -/
def subtypeProdEquivProd {α β} {p : α → Prop} {q : β → Prop} :
    { c : α × β // p c.1 ∧ q c.2 } ≃ { a // p a } × { b // q b } where
  toFun := fun x => ⟨⟨x.1.1, x.2.1⟩, ⟨x.1.2, x.2.2⟩⟩
  invFun := fun x => ⟨⟨x.1.1, x.2.1⟩, ⟨x.1.2, x.2.2⟩⟩
  left_inv := fun ⟨⟨_, _⟩, ⟨_, _⟩⟩ => rfl
  right_inv := fun ⟨⟨_, _⟩, ⟨_, _⟩⟩ => rfl


/-- A subtype of a `Prod` that depends only on the first component is equivalent to the
corresponding subtype of the first type times the second type. -/
def prodSubtypeFstEquivSubtypeProd {α β} {p : α → Prop} :
    {s : α × β // p s.1} ≃ {a // p a} × β where
  toFun x := ⟨⟨x.1.1, x.2⟩, x.1.2⟩
  invFun x := ⟨⟨x.1.1, x.2⟩, x.1.2⟩
  left_inv _ := rfl
  right_inv _ := rfl


/-- A subtype of a `Prod` is equivalent to a sigma type whose fibers are subtypes. -/
def subtypeProdEquivSigmaSubtype {α β} (p : α → β → Prop) :
    { x : α × β // p x.1 x.2 } ≃ Σa, { b : β // p a b } where
  toFun x := ⟨x.1.1, x.1.2, x.property⟩
  invFun x := ⟨⟨x.1, x.2⟩, x.2.property⟩
                   /-
                     α✝ : Sort u_1
                     α₁ : Sort u_2
                     α₂ : Sort u_3
                     β✝ : Sort u_4
                     β₁ : Sort u_5
                     β₂ : Sort u_6
                     γ : Sort u_7
                     δ : Sort u_8
                     α : Type ?u.79392
                     β : Type ?u.79391
                     p : α → β → Prop
                     x : Subtype fun x => p x.1 x.2
                     ⊢ Eq ((fun x => ⟨{ fst := x.fst, snd := ↑x.snd }, ⋯⟩) ((fun x => ⟨(↑x).1, ⟨(↑x …
                   -/
                           /-
                             🎉 no goals
                           -/
  left_inv x := by ext <;> rfl
                           /-
                             🎉 no goals
                           -/
  right_inv := fun ⟨_, _, _⟩ => rfl


/-- The type `∀ (i : α), β i` can be split as a product by separating the indices in `α`
depending on whether they satisfy a predicate `p` or not. -/
@[simps]
def piEquivPiSubtypeProd {α : Type*} (p : α → Prop) (β : α → Type*) [DecidablePred p] :
    (∀ i : α, β i) ≃ (∀ i : { x // p x }, β i) × ∀ i : { x // ¬p x }, β i where
  toFun f := (fun x => f x, fun x => f x)
  invFun f x := if h : p x then f.1 ⟨x, h⟩ else f.2 ⟨x, h⟩
  right_inv := by
    /-
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α : Type u_9
      p : α → Prop
      β : α → Type u_10
      inst✝ : DecidablePred p
      ⊢ Function.RightInverse (fun f x => dite (p x) (fun h => f.1 ⟨x, h⟩) fun h =>  …
    -/
    rintro ⟨f, g⟩
    /-
      case mk
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α : Type u_9
      p : α → Prop
      β : α → Type u_10
      inst✝ : DecidablePred p
      f : (i : Subtype fun x => p x) → β ↑i
      g : (i : Subtype fun x => Not (p x)) → β ↑i
      ⊢ Eq ((fun f => { fst := fun x => f ↑x, snd := fun x => f ↑x }) ((fun f x => d …
    -/
    ext1 <;>
        /-
          case mk.fst
          α✝ : Sort u_1
          α₁ : Sort u_2
          α₂ : Sort u_3
          β✝ : Sort u_4
          β₁ : Sort u_5
          β₂ : Sort u_6
          γ : Sort u_7
          δ : Sort u_8
          α : Type u_9
          p : α → Prop
          β : α → Type u_10
          inst✝ : DecidablePred p
          f : (i : Subtype fun x => p x) → β ↑i
          g : (i : Subtype fun x => Not (p x)) → β ↑i
          ⊢ Eq ((fun f => { fst := fun x => f ↑x, snd := fun x => f ↑x }) ((fun f x => d …
        -/
        /-
          case mk.fst.h
          α✝ : Sort u_1
          α₁ : Sort u_2
          α₂ : Sort u_3
          β✝ : Sort u_4
          β₁ : Sort u_5
          β₂ : Sort u_6
          γ : Sort u_7
          δ : Sort u_8
          α : Type u_9
          p : α → Prop
          β : α → Type u_10
          inst✝ : DecidablePred p
          f : (i : Subtype fun x => p x) → β ↑i
          g : (i : Subtype fun x => Not (p x)) → β ↑i
          y : Subtype fun x => p x
          ⊢ Eq (((fun f => { fst := fun x => f ↑x, snd := fun x => f ↑x }) ((fun f x =>  …
        -/
    /-
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α : Type u_9
      p : α → Prop
      β : α → Type u_10
      inst✝ : DecidablePred p
      f : (i : α) → β i
      ⊢ Eq ((fun f x => dite (p x) (fun h => f.1 ⟨x, h⟩) fun h => f.2 ⟨x, h⟩) ((fun  …
    -/
        /-
          case mk.fst.h.mk
          α✝ : Sort u_1
          α₁ : Sort u_2
          α₂ : Sort u_3
          β✝ : Sort u_4
          β₁ : Sort u_5
          β₂ : Sort u_6
          γ : Sort u_7
          δ : Sort u_8
          α : Type u_9
          p : α → Prop
          β : α → Type u_10
          inst✝ : DecidablePred p
          f : (i : Subtype fun x => p x) → β ↑i
          g : (i : Subtype fun x => Not (p x)) → β ↑i
          val : α
          property : p val
          ⊢ Eq (((fun f => { fst := fun x => f ↑x, snd := fun x => f ↑x }) ((fun f x =>  …
        -/
    /-
      case h
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α : Type u_9
      p : α → Prop
      β : α → Type u_10
      inst✝ : DecidablePred p
      f : (i : α) → β i
      x : α
      ⊢ Eq ((fun f x => dite (p x) (fun h => f.1 ⟨x, h⟩) fun h => f.2 ⟨x, h⟩) ((fun  …
    -/
        /-
          🎉 no goals
        -/
        /-
          case pos
          α✝ : Sort u_1
          α₁ : Sort u_2
          α₂ : Sort u_3
          β✝ : Sort u_4
          β₁ : Sort u_5
          β₂ : Sort u_6
          γ : Sort u_7
          δ : Sort u_8
          α : Type u_9
          p : α → Prop
          β : α → Type u_10
          inst✝ : DecidablePred p
          f : (i : α) → β i
          x : α
          h : p x
          ⊢ Eq ((fun f x => dite (p x) (fun h => f.1 ⟨x, h⟩) fun h => f.2 ⟨x, h⟩) ((fun  …
        -/
        /-
          🎉 no goals
        -/
        rcases y with ⟨val, property⟩
        /-
          🎉 no goals
        -/
        /-
          case mk.snd.h.mk
          α✝ : Sort u_1
          α₁ : Sort u_2
          α₂ : Sort u_3
          β✝ : Sort u_4
          β₁ : Sort u_5
          β₂ : Sort u_6
          γ : Sort u_7
          δ : Sort u_8
          α : Type u_9
          p : α → Prop
          β : α → Type u_10
          inst✝ : DecidablePred p
          f : (i : Subtype fun x => p x) → β ↑i
          g : (i : Subtype fun x => Not (p x)) → β ↑i
          val : α
          property : Not (p val)
          ⊢ Eq (((fun f => { fst := fun x => f ↑x, snd := fun x => f ↑x }) ((fun f x =>  …
        -/
        simp only [property, dif_pos, dif_neg, not_false_iff, Subtype.coe_mk]
        /-
          🎉 no goals
        -/
  left_inv f := by
    ext x
    by_cases h : p x <;>
      · simp only [h, dif_neg, dif_pos, not_false_iff]


/-- A product of types can be split as the binary product of one of the types and the product
  of all the remaining types. -/
@[simps]
def piSplitAt {α : Type*} [DecidableEq α] (i : α) (β : α → Type*) :
    (∀ j, β j) ≃ β i × ∀ j : { j // j ≠ i }, β j where
  toFun f := ⟨f i, fun j => f j⟩
  invFun f j := if h : j = i then h.symm.rec f.1 else f.2 ⟨j, h⟩
  right_inv f := by
    /-
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α : Type u_9
      inst✝ : DecidableEq α
      i : α
      β : α → Type u_10
      f : Prod (β i) ((j : Subtype fun j => Ne j i) → β ↑j)
      ⊢ Eq ((fun f => { fst := f i, snd := fun j => f ↑j }) ((fun f j => dite (Eq j  …
    -/
    ext x
    /-
      case fst
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α : Type u_9
      inst✝ : DecidableEq α
      i : α
      β : α → Type u_10
      f : Prod (β i) ((j : Subtype fun j => Ne j i) → β ↑j)
      ⊢ Eq ((fun f => { fst := f i, snd := fun j => f ↑j }) ((fun f j => dite (Eq j  …
    -/
    /-
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α : Type u_9
      inst✝ : DecidableEq α
      i : α
      β : α → Type u_10
      f : (j : α) → β j
      ⊢ Eq ((fun f j => dite (Eq j i) (fun h => Eq.rec f.1 ⋯) fun h => f.2 ⟨j, h⟩) ( …
    -/
    exacts [dif_pos rfl, (dif_neg x.2).trans (by cases x; rfl)]
    /-
      case h
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α : Type u_9
      inst✝ : DecidableEq α
      i : α
      β : α → Type u_10
      f : (j : α) → β j
      x : α
      ⊢ Eq ((fun f j => dite (Eq j i) (fun h => Eq.rec f.1 ⋯) fun h => f.2 ⟨j, h⟩) ( …
    -/
    /-
      🎉 no goals
    -/
    /-
      case h
      α✝ : Sort u_1
      α₁ : Sort u_2
      α₂ : Sort u_3
      β✝ : Sort u_4
      β₁ : Sort u_5
      β₂ : Sort u_6
      γ : Sort u_7
      δ : Sort u_8
      α : Type u_9
      inst✝ : DecidableEq α
      i : α
      β : α → Type u_10
      f : (j : α) → β j
      x : α
      ⊢ Eq (dite (Eq x i) (fun h => Eq.rec (f i) ⋯) fun h => f x) (f x)
    -/
  left_inv f := by
      /-
        case pos
        α✝ : Sort u_1
        α₁ : Sort u_2
        α₂ : Sort u_3
        β✝ : Sort u_4
        β₁ : Sort u_5
        β₂ : Sort u_6
        γ : Sort u_7
        δ : Sort u_8
        α : Type u_9
        inst✝ : DecidableEq α
        i : α
        β : α → Type u_10
        f : (j : α) → β j
        x : α
        h : Eq x i
        ⊢ Eq (Eq.rec (f i) ⋯) (f x)
      -/
    ext x
               /-
                 🎉 no goals
               -/
      /-
        case neg
        α✝ : Sort u_1
        α₁ : Sort u_2
        α₂ : Sort u_3
        β✝ : Sort u_4
        β₁ : Sort u_5
        β₂ : Sort u_6
        γ : Sort u_7
        δ : Sort u_8
        α : Type u_9
        inst✝ : DecidableEq α
        i : α
        β : α → Type u_10
        f : (j : α) → β j
        x : α
        h : Not (Eq x i)
        ⊢ Eq (f x) (f x)
      -/
    dsimp only
      /-
        🎉 no goals
      -/
    split_ifs with h
    · subst h; rfl
    · rfl


/-- A product of copies of a type can be split as the binary product of one copy and the product
  of all the remaining copies. -/
@[simps!]
def funSplitAt {α : Type*} [DecidableEq α] (i : α) (β : Type*) :
    (α → β) ≃ β × ({ j // j ≠ i } → β) :=
  piSplitAt i _


/-- The type of all functions `X → Y` with prescribed values for all `x' ≠ x`
is equivalent to the codomain `Y`. -/
def subtypeEquivCodomain (f : { x' // x' ≠ x } → Y) :
    { g : X → Y // g ∘ (↑) = f } ≃ Y :=
  (subtypePreimage _ f).trans <|
    @funUnique { x' // ¬x' ≠ x } _ <|
      show Unique { x' // ¬x' ≠ x } from
        @Equiv.unique _ _
          (show Unique { x' // x' = x } from {
            default := ⟨x, rfl⟩, uniq := fun ⟨_, h⟩ => Subtype.val_injective h })
          (subtypeEquivRight fun _ => not_not)


@[simp]
theorem coe_subtypeEquivCodomain (f : { x' // x' ≠ x } → Y) :
    (subtypeEquivCodomain f : _ → Y) =
      fun g : { g : X → Y // g ∘ (↑) = f } => (g : X → Y) x :=
  rfl


@[simp]
theorem subtypeEquivCodomain_apply (f : { x' // x' ≠ x } → Y) (g) :
    subtypeEquivCodomain f g = (g : X → Y) x :=
  rfl


theorem coe_subtypeEquivCodomain_symm (f : { x' // x' ≠ x } → Y) :
    ((subtypeEquivCodomain f).symm : Y → _) = fun y =>
      ⟨fun x' => if h : x' ≠ x then f ⟨x', h⟩ else y, by
        /-
          α : Sort u_1
          α₁ : Sort u_2
          α₂ : Sort u_3
          β : Sort u_4
          β₁ : Sort u_5
          β₂ : Sort u_6
          γ : Sort u_7
          δ : Sort u_8
          X : Sort u_9
          Y : Sort u_10
          inst✝ : DecidableEq X
          x : X
          f : (Subtype fun x' => Ne x' x) → Y
          y : Y
          ⊢ Eq (Function.comp (fun x' => dite (Ne x' x) (fun h => f ⟨x', h⟩) fun h => y) …
        -/
        funext x'
        /-
          case h
          α : Sort u_1
          α₁ : Sort u_2
          α₂ : Sort u_3
          β : Sort u_4
          β₁ : Sort u_5
          β₂ : Sort u_6
          γ : Sort u_7
          δ : Sort u_8
          X : Sort u_9
          Y : Sort u_10
          inst✝ : DecidableEq X
          x : X
          f : (Subtype fun x' => Ne x' x) → Y
          y : Y
          x' : Subtype fun x' => Ne x' x
          ⊢ Eq (Function.comp (fun x' => dite (Ne x' x) (fun h => f ⟨x', h⟩) fun h => y) …
        -/
        simp only [ne_eq, dite_not, comp_apply, Subtype.coe_eta, dite_eq_ite, ite_eq_right_iff]
        /-
          case h
          α : Sort u_1
          α₁ : Sort u_2
          α₂ : Sort u_3
          β : Sort u_4
          β₁ : Sort u_5
          β₂ : Sort u_6
          γ : Sort u_7
          δ : Sort u_8
          X : Sort u_9
          Y : Sort u_10
          inst✝ : DecidableEq X
          x : X
          f : (Subtype fun x' => Ne x' x) → Y
          y : Y
          x' : Subtype fun x' => Ne x' x
          ⊢ Eq (↑x') x → Eq y (f x')
        -/
        intro w
        /-
          case h
          α : Sort u_1
          α₁ : Sort u_2
          α₂ : Sort u_3
          β : Sort u_4
          β₁ : Sort u_5
          β₂ : Sort u_6
          γ : Sort u_7
          δ : Sort u_8
          X : Sort u_9
          Y : Sort u_10
          inst✝ : DecidableEq X
          x : X
          f : (Subtype fun x' => Ne x' x) → Y
          y : Y
          x' : Subtype fun x' => Ne x' x
          w : Eq (↑x') x
          ⊢ Eq y (f x')
        -/
        exfalso
        /-
          case h
          α : Sort u_1
          α₁ : Sort u_2
          α₂ : Sort u_3
          β : Sort u_4
          β₁ : Sort u_5
          β₂ : Sort u_6
          γ : Sort u_7
          δ : Sort u_8
          X : Sort u_9
          Y : Sort u_10
          inst✝ : DecidableEq X
          x : X
          f : (Subtype fun x' => Ne x' x) → Y
          y : Y
          x' : Subtype fun x' => Ne x' x
          w : Eq (↑x') x
          ⊢ False
        -/
        exact x'.property w⟩ :=
        /-
          🎉 no goals
        -/
  rfl


@[simp]
theorem subtypeEquivCodomain_symm_apply (f : { x' // x' ≠ x } → Y) (y : Y) (x' : X) :
    ((subtypeEquivCodomain f).symm y : X → Y) x' = if h : x' ≠ x then f ⟨x', h⟩ else y :=
  rfl


theorem subtypeEquivCodomain_symm_apply_eq (f : { x' // x' ≠ x } → Y) (y : Y) :
    ((subtypeEquivCodomain f).symm y : X → Y) x = y :=
  dif_neg (not_not.mpr rfl)


theorem subtypeEquivCodomain_symm_apply_ne
    (f : { x' // x' ≠ x } → Y) (y : Y) (x' : X) (h : x' ≠ x) :
    ((subtypeEquivCodomain f).symm y : X → Y) x' = f ⟨x', h⟩ :=
  dif_pos h


instance : CanLift (α → β) (α ≃ β) (↑) Bijective where prf f hf := ⟨ofBijective f hf, rfl⟩


/-- Extend the domain of `e : Equiv.Perm α` to one that is over `β` via `f : α → Subtype p`,
where `p : β → Prop`, permuting only the `b : β` that satisfy `p b`.
This can be used to extend the domain across a function `f : α → β`,
keeping everything outside of `Set.range f` fixed. For this use-case `Equiv` given by `f` can
be constructed by `Equiv.of_leftInverse'` or `Equiv.of_leftInverse` when there is a known
inverse, or `Equiv.ofInjective` in the general case.
-/
def Perm.extendDomain : Perm β' :=
  (permCongr f e).subtypeCongr (Equiv.refl _)


@[simp]
theorem Perm.extendDomain_apply_image (a : α') : e.extendDomain f (f a) = f (e a) := by
  /-
    α' : Type u_9
    β' : Type u_10
    e : Equiv.Perm α'
    p : β' → Prop
    inst✝ : DecidablePred p
    f : Equiv α' (Subtype p)
    a : α'
    ⊢ Eq ((e.extendDomain f) ↑(f a)) ↑(f (e a))
  -/
  simp [Perm.extendDomain]
  /-
    🎉 no goals
  -/


theorem Perm.extendDomain_apply_subtype {b : β'} (h : p b) :
    e.extendDomain f b = f (e (f.symm ⟨b, h⟩)) := by
  /-
    α' : Type u_9
    β' : Type u_10
    e : Equiv.Perm α'
    p : β' → Prop
    inst✝ : DecidablePred p
    f : Equiv α' (Subtype p)
    b : β'
    h : p b
    ⊢ Eq ((e.extendDomain f) b) ↑(f (e (f.symm ⟨b, h⟩)))
  -/
  simp [Perm.extendDomain, h]
  /-
    🎉 no goals
  -/


theorem Perm.extendDomain_apply_not_subtype {b : β'} (h : ¬p b) : e.extendDomain f b = b := by
  /-
    α' : Type u_9
    β' : Type u_10
    e : Equiv.Perm α'
    p : β' → Prop
    inst✝ : DecidablePred p
    f : Equiv α' (Subtype p)
    b : β'
    h : Not (p b)
    ⊢ Eq ((e.extendDomain f) b) b
  -/
  simp [Perm.extendDomain, h]
  /-
    🎉 no goals
  -/


@[simp]
theorem Perm.extendDomain_refl : Perm.extendDomain (Equiv.refl _) f = Equiv.refl _ := by
  /-
    α' : Type u_9
    β' : Type u_10
    p : β' → Prop
    inst✝ : DecidablePred p
    f : Equiv α' (Subtype p)
    ⊢ Eq (Equiv.Perm.extendDomain (Equiv.refl α') f) (Equiv.refl β')
  -/
  simp [Perm.extendDomain]
  /-
    🎉 no goals
  -/


@[simp]
theorem Perm.extendDomain_symm : (e.extendDomain f).symm = Perm.extendDomain e.symm f :=
  rfl


theorem Perm.extendDomain_trans (e e' : Perm α') :
    (e.extendDomain f).trans (e'.extendDomain f) = Perm.extendDomain (e.trans e') f := by
  /-
    α' : Type u_9
    β' : Type u_10
    p : β' → Prop
    inst✝ : DecidablePred p
    f : Equiv α' (Subtype p)
    e e' : Equiv.Perm α'
    ⊢ Eq (Equiv.trans (e.extendDomain f) (e'.extendDomain f)) (Equiv.Perm.extendDo …
  -/
  simp [Perm.extendDomain, permCongr_trans]
  /-
    🎉 no goals
  -/


/-- Subtype of the quotient is equivalent to the quotient of the subtype. Let `α` be a setoid with
equivalence relation `~`. Let `p₂` be a predicate on the quotient type `α/~`, and `p₁` be the lift
of this predicate to `α`: `p₁ a ↔ p₂ ⟦a⟧`. Let `~₂` be the restriction of `~` to `{x // p₁ x}`.
Then `{x // p₂ x}` is equivalent to the quotient of `{x // p₁ x}` by `~₂`. -/
def subtypeQuotientEquivQuotientSubtype (p₁ : α → Prop) {s₁ : Setoid α} {s₂ : Setoid (Subtype p₁)}
    (p₂ : Quotient s₁ → Prop) (hp₂ : ∀ a, p₁ a ↔ p₂ ⟦a⟧)
    (h : ∀ x y : Subtype p₁, s₂.r x y ↔ s₁.r x y) : {x // p₂ x} ≃ Quotient s₂ where
  toFun a :=
    Quotient.hrecOn a.1 (fun a h => ⟦⟨a, (hp₂ _).2 h⟩⟧)
                                  /-
                                    α : Sort u_1
                                    α₁ : Sort u_2
                                    α₂ : Sort u_3
                                    β : Sort u_4
                                    β₁ : Sort u_5
                                    β₂ : Sort u_6
                                    γ : Sort u_7
                                    δ : Sort u_8
                                    p₁ : α → Prop
                                    s₁ : Setoid α
                                    s₂ : Setoid (Subtype p₁)
                                    p₂ : Quotient s₁ → Prop
                                    hp₂ : ∀ (a : α), Iff (p₁ a) (p₂ (Quotient.mk s₁ a))
                                    h : ∀ (x y : Subtype p₁), Iff (s₂ x y) (s₁ ↑x ↑y)
                                    a✝ : Subtype fun x => p₂ x
                                    a b : α
                                    hab : HasEquiv.Equiv a b
                                    ⊢ Eq (p₂ (Quotient.mk s₁ a)) (p₂ (Quotient.mk s₁ b))
                                  -/
      (fun a b hab => hfunext (by rw [Quotient.sound hab]) fun _ _ _ =>
                                  /-
                                    🎉 no goals
                                  -/
        heq_of_eq (Quotient.sound ((h _ _).2 hab)))
      a.2
  invFun a :=
    Quotient.liftOn a (fun a => (⟨⟦a.1⟧, (hp₂ _).1 a.2⟩ : { x // p₂ x })) fun _ _ hab =>
      Subtype.ext_val (Quotient.sound ((h _ _).1 hab))
                 /-
                   α : Sort u_1
                   α₁ : Sort u_2
                   α₂ : Sort u_3
                   β : Sort u_4
                   β₁ : Sort u_5
                   β₂ : Sort u_6
                   γ : Sort u_7
                   δ : Sort u_8
                   p₁ : α → Prop
                   s₁ : Setoid α
                   s₂ : Setoid (Subtype p₁)
                   p₂ : Quotient s₁ → Prop
                   hp₂ : ∀ (a : α), Iff (p₁ a) (p₂ (Quotient.mk s₁ a))
                   h : ∀ (x y : Subtype p₁), Iff (s₂ x y) (s₁ ↑x ↑y)
                   ⊢ Function.LeftInverse (fun a => a.liftOn (fun a => ⟨Quotient.mk s₁ ↑a, ⋯⟩) ⋯) …
                 -/
  left_inv := by exact fun ⟨a, ha⟩ => Quotient.inductionOn a (fun b hb => rfl) ha
                 /-
                   🎉 no goals
                 -/
                    /-
                      α : Sort u_1
                      α₁ : Sort u_2
                      α₂ : Sort u_3
                      β : Sort u_4
                      β₁ : Sort u_5
                      β₂ : Sort u_6
                      γ : Sort u_7
                      δ : Sort u_8
                      p₁ : α → Prop
                      s₁ : Setoid α
                      s₂ : Setoid (Subtype p₁)
                      p₂ : Quotient s₁ → Prop
                      hp₂ : ∀ (a : α), Iff (p₁ a) (p₂ (Quotient.mk s₁ a))
                      h : ∀ (x y : Subtype p₁), Iff (s₂ x y) (s₁ ↑x ↑y)
                      a : Quotient s₂
                      ⊢ Eq ((fun a => Quotient.hrecOn (motive := fun x => p₂ x → Quotient s₂) (↑a) ( …
                    -/
  right_inv a := by exact Quotient.inductionOn a fun ⟨a, ha⟩ => rfl
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem subtypeQuotientEquivQuotientSubtype_mk (p₁ : α → Prop)
    [s₁ : Setoid α] [s₂ : Setoid (Subtype p₁)] (p₂ : Quotient s₁ → Prop) (hp₂ : ∀ a, p₁ a ↔ p₂ ⟦a⟧)
    (h : ∀ x y : Subtype p₁, s₂ x y ↔ (x : α) ≈ y)
    (x hx) : subtypeQuotientEquivQuotientSubtype p₁ p₂ hp₂ h ⟨⟦x⟧, hx⟩ = ⟦⟨x, (hp₂ _).2 hx⟩⟧ :=
  rfl


@[simp]
theorem subtypeQuotientEquivQuotientSubtype_symm_mk (p₁ : α → Prop)
    [s₁ : Setoid α] [s₂ : Setoid (Subtype p₁)] (p₂ : Quotient s₁ → Prop) (hp₂ : ∀ a, p₁ a ↔ p₂ ⟦a⟧)
    (h : ∀ x y : Subtype p₁, s₂ x y ↔ (x : α) ≈ y) (x) :
    (subtypeQuotientEquivQuotientSubtype p₁ p₂ hp₂ h).symm ⟦x⟧ = ⟨⟦x⟧, (hp₂ _).1 x.property⟩ :=
  rfl


/-- A helper function for `Equiv.swap`. -/
def swapCore (a b r : α) : α :=
  if r = a then b else if r = b then a else r


theorem swapCore_self (r a : α) : swapCore a a r = r := by
  /-
    α : Sort u_1
    inst✝ : DecidableEq α
    r a : α
    ⊢ Eq (Equiv.swapCore a a r) r
  -/
  unfold swapCore
  /-
    α : Sort u_1
    inst✝ : DecidableEq α
    r a : α
    ⊢ Eq (ite (Eq r a) a (ite (Eq r a) a r)) r
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp [*]
                /-
                  🎉 no goals
                -/


theorem swapCore_swapCore (r a b : α) : swapCore a b (swapCore a b r) = r := by
  /-
    α : Sort u_1
    inst✝ : DecidableEq α
    r a b : α
    ⊢ Eq (Equiv.swapCore a b (Equiv.swapCore a b r)) r
  -/
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
  unfold swapCore; split_ifs <;> cc
                                 /-
                                   🎉 no goals
                                 -/


theorem swapCore_comm (r a b : α) : swapCore a b r = swapCore b a r := by
  /-
    α : Sort u_1
    inst✝ : DecidableEq α
    r a b : α
    ⊢ Eq (Equiv.swapCore a b r) (Equiv.swapCore b a r)
  -/
  unfold swapCore
  -- Porting note: whatever solution works for `swapCore_swapCore` will work here too.
  /-
    α : Sort u_1
    inst✝ : DecidableEq α
    r a b : α
    ⊢ Eq (ite (Eq r a) b (ite (Eq r b) a r)) (ite (Eq r b) a (ite (Eq r a) b r))
  -/
                              /-
                                🎉 no goals
                              -/
                              /-
                                🎉 no goals
                              -/
  split_ifs with h₁ h₂ h₃ <;> try simp
                              /-
                                🎉 no goals
                              -/
    /-
      case pos
      α : Sort u_1
      inst✝ : DecidableEq α
      r a b : α
      h₁ : Eq r a
      h₂ : Eq r b
      ⊢ Eq b a
    -/
  · cases h₁; cases h₂; rfl
                        /-
                          🎉 no goals
                        -/


/-- `swap a b` is the permutation that swaps `a` and `b` and
  leaves other values as is. -/
def swap (a b : α) : Perm α :=
  ⟨swapCore a b, swapCore a b, fun r => swapCore_swapCore r a b,
    fun r => swapCore_swapCore r a b⟩


@[simp]
theorem swap_self (a : α) : swap a a = Equiv.refl _ :=
  ext fun r => swapCore_self r a


theorem swap_comm (a b : α) : swap a b = swap b a :=
  ext fun r => swapCore_comm r _ _


theorem swap_apply_def (a b x : α) : swap a b x = if x = a then b else if x = b then a else x :=
  rfl


@[simp]
theorem swap_apply_left (a b : α) : swap a b a = b :=
  if_pos rfl


@[simp]
theorem swap_apply_right (a b : α) : swap a b b = a := by
  /-
    α : Sort u_1
    inst✝ : DecidableEq α
    a b : α
    ⊢ Eq ((Equiv.swap a b) b) a
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : b = a <;> simp [swap_apply_def, h]
                         /-
                           🎉 no goals
                         -/


theorem swap_apply_of_ne_of_ne {a b x : α} : x ≠ a → x ≠ b → swap a b x = x := by
  /-
    α : Sort u_1
    inst✝ : DecidableEq α
    a b x : α
    ⊢ Ne x a → Ne x b → Eq ((Equiv.swap a b) x) x
  -/
  simp +contextual [swap_apply_def]
  /-
    🎉 no goals
  -/


theorem eq_or_eq_of_swap_apply_ne_self {a b x : α} (h : swap a b x ≠ x) : x = a ∨ x = b := by
  /-
    α : Sort u_1
    inst✝ : DecidableEq α
    a b x : α
    h : Ne ((Equiv.swap a b) x) x
    ⊢ Or (Eq x a) (Eq x b)
  -/
  contrapose! h
  /-
    α : Sort u_1
    inst✝ : DecidableEq α
    a b x : α
    h : And (Ne x a) (Ne x b)
    ⊢ Eq ((Equiv.swap a b) x) x
  -/
  exact swap_apply_of_ne_of_ne h.1 h.2
  /-
    🎉 no goals
  -/


@[simp]
theorem swap_swap (a b : α) : (swap a b).trans (swap a b) = Equiv.refl _ :=
  ext fun _ => swapCore_swapCore _ _ _


@[simp]
theorem symm_swap (a b : α) : (swap a b).symm = swap a b :=
  rfl


@[simp]
theorem swap_eq_refl_iff {x y : α} : swap x y = Equiv.refl _ ↔ x = y := by
  /-
    α : Sort u_1
    inst✝ : DecidableEq α
    x y : α
    ⊢ Iff (Eq (Equiv.swap x y) (Equiv.refl α)) (Eq x y)
  -/
  refine ⟨fun h => (Equiv.refl _).injective ?_, fun h => h ▸ swap_self _⟩
  /-
    α : Sort u_1
    inst✝ : DecidableEq α
    x y : α
    h : Eq (Equiv.swap x y) (Equiv.refl α)
    ⊢ Eq ((Equiv.refl α) x) ((Equiv.refl α) y)
  -/
  rw [← h, swap_apply_left, h, refl_apply]
  /-
    🎉 no goals
  -/


theorem swap_comp_apply {a b x : α} (π : Perm α) :
    π.trans (swap a b) x = if π x = a then b else if π x = b then a else π x := by
  /-
    α : Sort u_1
    inst✝ : DecidableEq α
    a b x : α
    π : Equiv.Perm α
    ⊢ Eq ((Equiv.trans π (Equiv.swap a b)) x) (ite (Eq (π x) a) b (ite (Eq (π x) b …
  -/
  cases π
  /-
    case mk
    α : Sort u_1
    inst✝ : DecidableEq α
    a b x : α
    toFun✝ invFun✝ : α → α
    left_inv✝ : Function.LeftInverse invFun✝ toFun✝
    right_inv✝ : Function.RightInverse invFun✝ toFun✝
    ⊢ Eq (({ toFun := toFun✝, invFun := invFun✝, left_inv := left_inv✝, right_inv  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem swap_eq_update (i j : α) : (Equiv.swap i j : α → α) = update (update id j i) i j :=
                     /-
                       α : Sort u_1
                       inst✝ : DecidableEq α
                       i j x : α
                       ⊢ Eq ((Equiv.swap i j) x) (Function.update (Function.update id j i) i j x)
                     -/
  funext fun x => by rw [update_apply _ i j, update_apply _ j i, Equiv.swap_apply_def, id]
                     /-
                       🎉 no goals
                     -/


theorem comp_swap_eq_update (i j : α) (f : α → β) :
    f ∘ Equiv.swap i j = update (update f j (f i)) i (f j) := by
  /-
    α : Sort u_1
    β : Sort u_4
    inst✝ : DecidableEq α
    i j : α
    f : α → β
    ⊢ Eq (Function.comp f ⇑(Equiv.swap i j)) (Function.update (Function.update f j …
  -/
  rw [swap_eq_update, comp_update, comp_update, comp_id]
  /-
    🎉 no goals
  -/


@[simp]
theorem symm_trans_swap_trans [DecidableEq β] (a b : α) (e : α ≃ β) :
    (e.symm.trans (swap a b)).trans e = swap (e a) (e b) :=
  Equiv.ext fun x => by
    have : ∀ a, e.symm x = a ↔ x = e a := fun a => by
      rw [@eq_comm _ (e.symm x)]
      constructor <;> intros <;> simp_all
    /-
      α : Sort u_1
      β : Sort u_4
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      a b : α
      e : Equiv α β
      x : β
      this : ∀ (a : α), Iff (Eq (e.symm x) a) (Eq x (e a))
      ⊢ Eq (((e.symm.trans (Equiv.swap a b)).trans e) x) ((Equiv.swap (e a) (e b)) x)
    -/
    simp only [trans_apply, swap_apply_def, this]
    /-
      α : Sort u_1
      β : Sort u_4
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      a b : α
      e : Equiv α β
      x : β
      this : ∀ (a : α), Iff (Eq (e.symm x) a) (Eq x (e a))
      ⊢ Eq (e (ite (Eq x (e a)) b (ite (Eq x (e b)) a (e.symm x)))) (ite (Eq x (e a) …
    -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
    split_ifs <;> simp
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem trans_swap_trans_symm [DecidableEq β] (a b : β) (e : α ≃ β) :
    (e.trans (swap a b)).trans e.symm = swap (e.symm a) (e.symm b) :=
  symm_trans_swap_trans a b e.symm


@[simp]
theorem swap_apply_self (i j a : α) : swap i j (swap i j a) = a := by
  /-
    α : Sort u_1
    inst✝ : DecidableEq α
    i j a : α
    ⊢ Eq ((Equiv.swap i j) ((Equiv.swap i j) a)) a
  -/
  rw [← Equiv.trans_apply, Equiv.swap_swap, Equiv.refl_apply]
  /-
    🎉 no goals
  -/


/-- A function is invariant to a swap if it is equal at both elements -/
theorem apply_swap_eq_self {v : α → β} {i j : α} (hv : v i = v j) (k : α) :
    v (swap i j k) = v k := by
  /-
    α : Sort u_1
    β : Sort u_4
    inst✝ : DecidableEq α
    v : α → β
    i j : α
    hv : Eq (v i) (v j)
    k : α
    ⊢ Eq (v ((Equiv.swap i j) k)) (v k)
  -/
  by_cases hi : k = i
    /-
      case pos
      α : Sort u_1
      β : Sort u_4
      inst✝ : DecidableEq α
      v : α → β
      i j : α
      hv : Eq (v i) (v j)
      k : α
      hi : Eq k i
      ⊢ Eq (v ((Equiv.swap i j) k)) (v k)
    -/
  · rw [hi, swap_apply_left, hv]
    /-
      🎉 no goals
    -/

  /-
    case neg
    α : Sort u_1
    β : Sort u_4
    inst✝ : DecidableEq α
    v : α → β
    i j : α
    hv : Eq (v i) (v j)
    k : α
    hi : Not (Eq k i)
    ⊢ Eq (v ((Equiv.swap i j) k)) (v k)
  -/
  by_cases hj : k = j
    /-
      case pos
      α : Sort u_1
      β : Sort u_4
      inst✝ : DecidableEq α
      v : α → β
      i j : α
      hv : Eq (v i) (v j)
      k : α
      hi : Not (Eq k i)
      hj : Eq k j
      ⊢ Eq (v ((Equiv.swap i j) k)) (v k)
    -/
  · rw [hj, swap_apply_right, hv]
    /-
      🎉 no goals
    -/

  /-
    case neg
    α : Sort u_1
    β : Sort u_4
    inst✝ : DecidableEq α
    v : α → β
    i j : α
    hv : Eq (v i) (v j)
    k : α
    hi : Not (Eq k i)
    hj : Not (Eq k j)
    ⊢ Eq (v ((Equiv.swap i j) k)) (v k)
  -/
  rw [swap_apply_of_ne_of_ne hi hj]
  /-
    🎉 no goals
  -/


theorem swap_apply_eq_iff {x y z w : α} : swap x y z = w ↔ z = swap x y w := by
  /-
    α : Sort u_1
    inst✝ : DecidableEq α
    x y z w : α
    ⊢ Iff (Eq ((Equiv.swap x y) z) w) (Eq z ((Equiv.swap x y) w))
  -/
  rw [apply_eq_iff_eq_symm_apply, symm_swap]
  /-
    🎉 no goals
  -/


theorem swap_apply_ne_self_iff {a b x : α} : swap a b x ≠ x ↔ a ≠ b ∧ (x = a ∨ x = b) := by
  /-
    α : Sort u_1
    inst✝ : DecidableEq α
    a b x : α
    ⊢ Iff (Ne ((Equiv.swap a b) x) x) (And (Ne a b) (Or (Eq x a) (Eq x b)))
  -/
  by_cases hab : a = b
    /-
      case pos
      α : Sort u_1
      inst✝ : DecidableEq α
      a b x : α
      hab : Eq a b
      ⊢ Iff (Ne ((Equiv.swap a b) x) x) (And (Ne a b) (Or (Eq x a) (Eq x b)))
    -/
  · simp [hab]
    /-
      🎉 no goals
    -/

  /-
    case neg
    α : Sort u_1
    inst✝ : DecidableEq α
    a b x : α
    hab : Not (Eq a b)
    ⊢ Iff (Ne ((Equiv.swap a b) x) x) (And (Ne a b) (Or (Eq x a) (Eq x b)))
  -/
  by_cases hax : x = a
    /-
      case pos
      α : Sort u_1
      inst✝ : DecidableEq α
      a b x : α
      hab : Not (Eq a b)
      hax : Eq x a
      ⊢ Iff (Ne ((Equiv.swap a b) x) x) (And (Ne a b) (Or (Eq x a) (Eq x b)))
    -/
  · simp [hax, eq_comm]
    /-
      🎉 no goals
    -/

  /-
    case neg
    α : Sort u_1
    inst✝ : DecidableEq α
    a b x : α
    hab : Not (Eq a b)
    hax : Not (Eq x a)
    ⊢ Iff (Ne ((Equiv.swap a b) x) x) (And (Ne a b) (Or (Eq x a) (Eq x b)))
  -/
  by_cases hbx : x = b
    /-
      case pos
      α : Sort u_1
      inst✝ : DecidableEq α
      a b x : α
      hab : Not (Eq a b)
      hax : Not (Eq x a)
      hbx : Eq x b
      ⊢ Iff (Ne ((Equiv.swap a b) x) x) (And (Ne a b) (Or (Eq x a) (Eq x b)))
    -/
  · simp [hbx]
    /-
      🎉 no goals
    -/

  /-
    case neg
    α : Sort u_1
    inst✝ : DecidableEq α
    a b x : α
    hab : Not (Eq a b)
    hax : Not (Eq x a)
    hbx : Not (Eq x b)
    ⊢ Iff (Ne ((Equiv.swap a b) x) x) (And (Ne a b) (Or (Eq x a) (Eq x b)))
  -/
  simp [hab, hax, hbx, swap_apply_of_ne_of_ne]
  /-
    🎉 no goals
  -/


@[simp]
theorem sumCongr_swap_refl {α β : Sort _} [DecidableEq α] [DecidableEq β] (i j : α) :
    Equiv.Perm.sumCongr (Equiv.swap i j) (Equiv.refl β) = Equiv.swap (Sum.inl i) (Sum.inl j) := by
  /-
    α : Type u_9
    β : Type u_10
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    i j : α
    ⊢ Eq ((Equiv.swap i j).sumCongr (Equiv.refl β)) (Equiv.swap (Sum.inl i) (Sum.i …
  -/
  ext x
  /-
    case H
    α : Type u_9
    β : Type u_10
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    i j : α
    x : Sum α β
    ⊢ Eq (((Equiv.swap i j).sumCongr (Equiv.refl β)) x) ((Equiv.swap (Sum.inl i) ( …
  -/
  cases x
  · simp only [Equiv.sumCongr_apply, Sum.map, coe_refl, comp_id, Sum.elim_inl, comp_apply,
      swap_apply_def, Sum.inl.injEq]
    /-
      case H.inl
      α : Type u_9
      β : Type u_10
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      i j val✝ : α
      ⊢ Eq (Sum.inl (ite (Eq val✝ i) j (ite (Eq val✝ j) i val✝))) (ite (Eq val✝ i) ( …
    -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
    split_ifs <;> rfl
                  /-
                    🎉 no goals
                  -/
    /-
      case H.inr
      α : Type u_9
      β : Type u_10
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      i j : α
      val✝ : β
      ⊢ Eq (((Equiv.swap i j).sumCongr (Equiv.refl β)) (Sum.inr val✝)) ((Equiv.swap  …
    -/
  · simp [Sum.map, swap_apply_of_ne_of_ne]
    /-
      🎉 no goals
    -/


@[simp]
theorem sumCongr_refl_swap {α β : Sort _} [DecidableEq α] [DecidableEq β] (i j : β) :
    Equiv.Perm.sumCongr (Equiv.refl α) (Equiv.swap i j) = Equiv.swap (Sum.inr i) (Sum.inr j) := by
  /-
    α : Type u_9
    β : Type u_10
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    i j : β
    ⊢ Eq (Equiv.Perm.sumCongr (Equiv.refl α) (Equiv.swap i j)) (Equiv.swap (Sum.in …
  -/
  ext x
  /-
    case H
    α : Type u_9
    β : Type u_10
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    i j : β
    x : Sum α β
    ⊢ Eq ((Equiv.Perm.sumCongr (Equiv.refl α) (Equiv.swap i j)) x) ((Equiv.swap (S …
  -/
  cases x
    /-
      case H.inl
      α : Type u_9
      β : Type u_10
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      i j : β
      val✝ : α
      ⊢ Eq ((Equiv.Perm.sumCongr (Equiv.refl α) (Equiv.swap i j)) (Sum.inl val✝)) (( …
    -/
  · simp [Sum.map, swap_apply_of_ne_of_ne]
    /-
      🎉 no goals
    -/

  · simp only [Equiv.sumCongr_apply, Sum.map, coe_refl, comp_id, Sum.elim_inr, comp_apply,
      swap_apply_def, Sum.inr.injEq]
    /-
      case H.inr
      α : Type u_9
      β : Type u_10
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      i j val✝ : β
      ⊢ Eq (Sum.inr (ite (Eq val✝ i) j (ite (Eq val✝ j) i val✝))) (ite (Eq val✝ i) ( …
    -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
    split_ifs <;> rfl
                  /-
                    🎉 no goals
                  -/


/-- Augment an equivalence with a prescribed mapping `f a = b` -/
def setValue (f : α ≃ β) (a : α) (b : β) : α ≃ β :=
  (swap a (f.symm b)).trans f


@[simp]
theorem setValue_eq (f : α ≃ β) (a : α) (b : β) : setValue f a b a = b := by
  /-
    α : Sort u_1
    β : Sort u_4
    inst✝ : DecidableEq α
    f : Equiv α β
    a : α
    b : β
    ⊢ Eq ((f.setValue a b) a) b
  -/
  simp [setValue, swap_apply_left]
  /-
    🎉 no goals
  -/


/-- Convert an involutive function `f` to a permutation with `toFun = invFun = f`. -/
def toPerm (f : α → α) (h : Involutive f) : Equiv.Perm α :=
  ⟨f, f, h.leftInverse, h.rightInverse⟩


@[simp]
theorem coe_toPerm {f : α → α} (h : Involutive f) : (h.toPerm f : α → α) = f :=
  rfl


@[simp]
theorem toPerm_symm {f : α → α} (h : Involutive f) : (h.toPerm f).symm = h.toPerm f :=
  rfl


theorem toPerm_involutive {f : α → α} (h : Involutive f) : Involutive (h.toPerm f) :=
  h


theorem symm_eq_self_of_involutive (f : Equiv.Perm α) (h : Involutive f) : f.symm = f :=
  DFunLike.coe_injective (h.leftInverse_iff.mp f.left_inv)


theorem PLift.eq_up_iff_down_eq {x : PLift α} {y : α} : x = PLift.up y ↔ x.down = y :=
  Equiv.plift.eq_symm_apply


theorem Function.Injective.map_swap [DecidableEq α] [DecidableEq β] {f : α → β}
    (hf : Function.Injective f) (x y z : α) :
    f (Equiv.swap x y z) = Equiv.swap (f x) (f y) (f z) := by
  /-
    α : Sort u_1
    β : Sort u_4
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → β
    hf : Function.Injective f
    x y z : α
    ⊢ Eq (f ((Equiv.swap x y) z)) ((Equiv.swap (f x) (f y)) (f z))
  -/
  conv_rhs => rw [Equiv.swap_apply_def]
  /-
    α : Sort u_1
    β : Sort u_4
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → β
    hf : Function.Injective f
    x y z : α
    ⊢ Eq (f ((Equiv.swap x y) z)) (ite (Eq (f z) (f x)) (f y) (ite (Eq (f z) (f y) …
  -/
  split_ifs with h₁ h₂
    /-
      case pos
      α : Sort u_1
      β : Sort u_4
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → β
      hf : Function.Injective f
      x y z : α
      h₁ : Eq (f z) (f x)
      ⊢ Eq (f ((Equiv.swap x y) z)) (f y)
    -/
  · rw [hf h₁, Equiv.swap_apply_left]
    /-
      🎉 no goals
    -/
    /-
      case pos
      α : Sort u_1
      β : Sort u_4
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → β
      hf : Function.Injective f
      x y z : α
      h₁ : Not (Eq (f z) (f x))
      h₂ : Eq (f z) (f y)
      ⊢ Eq (f ((Equiv.swap x y) z)) (f x)
    -/
  · rw [hf h₂, Equiv.swap_apply_right]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Sort u_1
      β : Sort u_4
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → β
      hf : Function.Injective f
      x y z : α
      h₁ : Not (Eq (f z) (f x))
      h₂ : Not (Eq (f z) (f y))
      ⊢ Eq (f ((Equiv.swap x y) z)) (f z)
    -/
  · rw [Equiv.swap_apply_of_ne_of_ne (mt (congr_arg f) h₁) (mt (congr_arg f) h₂)]
    /-
      🎉 no goals
    -/


/-- Transport dependent functions through an equivalence of the base space.
-/
@[simps apply, simps (config := .lemmasOnly) symm_apply]
def piCongrLeft' (P : α → Sort*) (e : α ≃ β) : (∀ a, P a) ≃ ∀ b, P (e.symm b) where
  toFun f x := f (e.symm x)
  invFun f x := (e.symm_apply_apply x).ndrec (f (e x))
  left_inv f := funext fun x =>
        /-
          α : Sort u_1
          α₁ : Sort u_2
          α₂ : Sort u_3
          β : Sort u_4
          β₁ : Sort u_5
          β₂ : Sort u_6
          γ : Sort u_7
          δ : Sort u_8
          P : α → Sort u_9
          e : Equiv α β
          f : (a : α) → P a
          x : α
          ⊢ ∀ {y : α} (h : Eq y x), Eq (Eq.ndrec (f y) h) (f x)
        -/
    (by rintro _ rfl; rfl : ∀ {y} (h : y = x), h.ndrec (f y) = f x) (e.symm_apply_apply x)
                      /-
                        🎉 no goals
                      -/
  right_inv f := funext fun x =>
        /-
          α : Sort u_1
          α₁ : Sort u_2
          α₂ : Sort u_3
          β : Sort u_4
          β₁ : Sort u_5
          β₂ : Sort u_6
          γ : Sort u_7
          δ : Sort u_8
          P : α → Sort u_9
          e : Equiv α β
          f : (b : β) → P (e.symm b)
          x : β
          ⊢ ∀ {y : β} (h : Eq y x), Eq (Eq.ndrec (f y) ⋯) (f x)
        -/
    (by rintro _ rfl; rfl : ∀ {y} (h : y = x), (congr_arg e.symm h).ndrec (f y) = f x)
                      /-
                        🎉 no goals
                      -/
      (e.apply_symm_apply x)


/-- This lemma is impractical to state in the dependent case. -/
@[simp]
theorem piCongrLeft'_symm (P : Sort*) (e : α ≃ β) :
                                                                     /-
                                                                       α : Sort u_1
                                                                       β : Sort u_4
                                                                       P : Sort u_9
                                                                       e : Equiv α β
                                                                       ⊢ Eq (Equiv.piCongrLeft' (fun x => P) e).symm (Equiv.piCongrLeft' (fun a => P) …
                                                                     -/
    (piCongrLeft' (fun _ => P) e).symm = piCongrLeft' _ e.symm := by ext; simp [piCongrLeft']
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


/-- Note: the "obvious" statement `(piCongrLeft' P e).symm g a = g (e a)` doesn't typecheck: the
LHS would have type `P a` while the RHS would have type `P (e.symm (e a))`. This lemma is a way
around it in the case where `a` is of the form `e.symm b`, so we can use `g b` instead of
`g (e (e.symm b))`. -/
@[simp]
lemma piCongrLeft'_symm_apply_apply (P : α → Sort*) (e : α ≃ β) (g : ∀ b, P (e.symm b)) (b : β) :
    (piCongrLeft' P e).symm g (e.symm b) = g b := by
  /-
    α : Sort u_1
    β : Sort u_4
    P : α → Sort u_9
    e : Equiv α β
    g : (b : β) → P (e.symm b)
    b : β
    ⊢ Eq ((Equiv.piCongrLeft' P e).symm g (e.symm b)) (g b)
  -/
  rw [piCongrLeft'_symm_apply, ← heq_iff_eq, rec_heq_iff_heq]
  /-
    α : Sort u_1
    β : Sort u_4
    P : α → Sort u_9
    e : Equiv α β
    g : (b : β) → P (e.symm b)
    b : β
    ⊢ HEq (g (e (e.symm b))) (g b)
  -/
  exact congr_arg_heq _ (e.apply_symm_apply _)
  /-
    🎉 no goals
  -/


/-- Transporting dependent functions through an equivalence of the base,
expressed as a "simplification".
-/
def piCongrLeft : (∀ a, P (e a)) ≃ ∀ b, P b :=
  (piCongrLeft' P e.symm).symm


/-- Note: the "obvious" statement `(piCongrLeft P e) f b = f (e.symm b)` doesn't typecheck: the
LHS would have type `P b` while the RHS would have type `P (e (e.symm b))`. For that reason,
we have to explicitly substitute along `e (e.symm b) = b` in the statement of this lemma. -/
@[simp]
lemma piCongrLeft_apply (f : ∀ a, P (e a)) (b : β) :
    (piCongrLeft P e) f b = e.apply_symm_apply b ▸ f (e.symm b) :=
  rfl


@[simp]
lemma piCongrLeft_symm_apply (g : ∀ b, P b) (a : α) :
    (piCongrLeft P e).symm g a = g (e a) :=
  piCongrLeft'_apply P e.symm g a


/-- Note: the "obvious" statement `(piCongrLeft P e) f b = f (e.symm b)` doesn't typecheck: the
LHS would have type `P b` while the RHS would have type `P (e (e.symm b))`. This lemma is a way
around it in the case where `b` is of the form `e a`, so we can use `f a` instead of
`f (e.symm (e a))`. -/
lemma piCongrLeft_apply_apply (f : ∀ a, P (e a)) (a : α) :
    (piCongrLeft P e) f (e a) = f a :=
  piCongrLeft'_symm_apply_apply P e.symm f a


lemma piCongrLeft_apply_eq_cast {P : β → Sort v} {e : α ≃ β}
    (f : (a : α) → P (e a)) (b : β) :
    piCongrLeft P e f b = cast (congr_arg P (e.apply_symm_apply b)) (f (e.symm b)) :=
  Eq.rec_eq_cast _ _


theorem piCongrLeft_sum_inl {ι ι' ι''} (π : ι'' → Type*) (e : ι ⊕ ι' ≃ ι'') (f : ∀ i, π (e (inl i)))
    (g : ∀ i, π (e (inr i))) (i : ι) :
    piCongrLeft π e (sumPiEquivProdPi (fun x => π (e x)) |>.symm (f, g)) (e (inl i)) = f i := by
  simp_rw [piCongrLeft_apply_eq_cast, sumPiEquivProdPi_symm_apply,
    sum_rec_congr _ _ _ (e.symm_apply_apply (inl i)), cast_cast, cast_eq]


theorem piCongrLeft_sum_inr {ι ι' ι''} (π : ι'' → Type*) (e : ι ⊕ ι' ≃ ι'') (f : ∀ i, π (e (inl i)))
    (g : ∀ i, π (e (inr i))) (j : ι') :
    piCongrLeft π e (sumPiEquivProdPi (fun x => π (e x)) |>.symm (f, g)) (e (inr j)) = g j := by
  simp_rw [piCongrLeft_apply_eq_cast, sumPiEquivProdPi_symm_apply,
    sum_rec_congr _ _ _ (e.symm_apply_apply (inr j)), cast_cast, cast_eq]


/-- Transport dependent functions through
an equivalence of the base spaces and a family
of equivalences of the matching fibers.
-/
def piCongr : (∀ a, W a) ≃ ∀ b, Z b :=
  (Equiv.piCongrRight h₂).trans (Equiv.piCongrLeft _ h₁)


@[simp]
theorem coe_piCongr_symm : ((h₁.piCongr h₂).symm :
    (∀ b, Z b) → ∀ a, W a) = fun f a => (h₂ a).symm (f (h₁ a)) :=
  rfl


theorem piCongr_symm_apply (f : ∀ b, Z b) :
    (h₁.piCongr h₂).symm f = fun a => (h₂ a).symm (f (h₁ a)) :=
  rfl


@[simp]
theorem piCongr_apply_apply (f : ∀ a, W a) (a : α) : h₁.piCongr h₂ f (h₁ a) = h₂ a (f a) := by
  /-
    α : Sort u_1
    β : Sort u_4
    W : α → Sort w
    Z : β → Sort z
    h₁ : Equiv α β
    h₂ : (a : α) → Equiv (W a) (Z (h₁ a))
    f : (a : α) → W a
    a : α
    ⊢ Eq ((h₁.piCongr h₂) f (h₁ a)) ((h₂ a) (f a))
  -/
  simp only [piCongr, piCongrRight, trans_apply, coe_fn_mk, piCongrLeft_apply_apply, Pi.map_apply]
  /-
    🎉 no goals
  -/


/-- Transport dependent functions through
an equivalence of the base spaces and a family
of equivalences of the matching fibres.
-/
def piCongr' : (∀ a, W a) ≃ ∀ b, Z b :=
  (piCongr h₁.symm fun b => (h₂ b).symm).symm


@[simp]
theorem coe_piCongr' :
    (h₁.piCongr' h₂ : (∀ a, W a) → ∀ b, Z b) = fun f b => h₂ b <| f <| h₁.symm b :=
  rfl


theorem piCongr'_apply (f : ∀ a, W a) : h₁.piCongr' h₂ f = fun b => h₂ b <| f <| h₁.symm b :=
  rfl


@[simp]
theorem piCongr'_symm_apply_symm_apply (f : ∀ b, Z b) (b : β) :
    (h₁.piCongr' h₂).symm f (h₁.symm b) = (h₂ b).symm (f b) := by
  /-
    α : Sort u_1
    β : Sort u_4
    W : α → Sort w
    Z : β → Sort z
    h₁ : Equiv α β
    h₂ : (b : β) → Equiv (W (h₁.symm b)) (Z b)
    f : (b : β) → Z b
    b : β
    ⊢ Eq ((h₁.piCongr' h₂).symm f (h₁.symm b)) ((h₂ b).symm (f b))
  -/
  simp [piCongr', piCongr_apply_apply]
  /-
    🎉 no goals
  -/


/-- Transport dependent functions through an equality of sets. -/
@[simps!] def piCongrSet {α} {W : α → Sort w} {s t : Set α} (h : s = t) :
    (∀ i : {i // i ∈ s}, W i) ≃ (∀ i : {i // i ∈ t}, W i) where
  toFun f i := f ⟨i, h ▸ i.2⟩
  invFun f i := f ⟨i, h.symm ▸ i.2⟩
  left_inv f := rfl
  right_inv f := rfl


                                                                             /-
                                                                               α₁ : Type u_9
                                                                               β₁ : Type u_10
                                                                               e : Equiv α₁ β₁
                                                                               f : α₁ → α₁
                                                                               x : α₁
                                                                               ⊢ Eq (e (f x)) (e.conj f (e x))
                                                                             -/
theorem semiconj_conj (f : α₁ → α₁) : Semiconj e f (e.conj f) := fun x => by simp
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


                                                                                /-
                                                                                  α₁ : Type u_9
                                                                                  β₁ : Type u_10
                                                                                  e : Equiv α₁ β₁
                                                                                  f : α₁ → α₁ → α₁
                                                                                  x y : α₁
                                                                                  ⊢ Eq (e (f x y)) ((e.arrowCongr e.conj) f (e x) (e y))
                                                                                -/
theorem semiconj₂_conj : Semiconj₂ e f (e.arrowCongr e.conj f) := fun x y => by simp [arrowCongr]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


instance [Std.Associative f] : Std.Associative (e.arrowCongr (e.arrowCongr e) f) :=
  (e.semiconj₂_conj f).isAssociative_right e.surjective


instance [Std.IdempotentOp f] : Std.IdempotentOp (e.arrowCongr (e.arrowCongr e) f) :=
  (e.semiconj₂_conj f).isIdempotent_right e.surjective


@[simp]
theorem ulift_symm_down {α} (x : α) : (Equiv.ulift.{u, v}.symm x).down = x :=
  rfl


theorem Function.Injective.swap_apply
    [DecidableEq α] [DecidableEq β] {f : α → β} (hf : Function.Injective f) (x y z : α) :
    Equiv.swap (f x) (f y) (f z) = f (Equiv.swap x y z) := by
  /-
    α : Sort u_1
    β : Sort u_4
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → β
    hf : Function.Injective f
    x y z : α
    ⊢ Eq ((Equiv.swap (f x) (f y)) (f z)) (f ((Equiv.swap x y) z))
  -/
  by_cases hx : z = x
    /-
      case pos
      α : Sort u_1
      β : Sort u_4
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → β
      hf : Function.Injective f
      x y z : α
      hx : Eq z x
      ⊢ Eq ((Equiv.swap (f x) (f y)) (f z)) (f ((Equiv.swap x y) z))
    -/
  · simp [hx]
    /-
      🎉 no goals
    -/

  /-
    case neg
    α : Sort u_1
    β : Sort u_4
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → β
    hf : Function.Injective f
    x y z : α
    hx : Not (Eq z x)
    ⊢ Eq ((Equiv.swap (f x) (f y)) (f z)) (f ((Equiv.swap x y) z))
  -/
  by_cases hy : z = y
    /-
      case pos
      α : Sort u_1
      β : Sort u_4
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → β
      hf : Function.Injective f
      x y z : α
      hx : Not (Eq z x)
      hy : Eq z y
      ⊢ Eq ((Equiv.swap (f x) (f y)) (f z)) (f ((Equiv.swap x y) z))
    -/
  · simp [hy]
    /-
      🎉 no goals
    -/

  /-
    case neg
    α : Sort u_1
    β : Sort u_4
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → β
    hf : Function.Injective f
    x y z : α
    hx : Not (Eq z x)
    hy : Not (Eq z y)
    ⊢ Eq ((Equiv.swap (f x) (f y)) (f z)) (f ((Equiv.swap x y) z))
  -/
  rw [Equiv.swap_apply_of_ne_of_ne hx hy, Equiv.swap_apply_of_ne_of_ne (hf.ne hx) (hf.ne hy)]
  /-
    🎉 no goals
  -/


theorem Function.Injective.swap_comp
    [DecidableEq α] [DecidableEq β] {f : α → β} (hf : Function.Injective f) (x y : α) :
    Equiv.swap (f x) (f y) ∘ f = f ∘ Equiv.swap x y :=
  funext fun _ => hf.swap_apply _ _ _


/-- If `α` is a subsingleton, then it is equivalent to `α × α`. -/
def subsingletonProdSelfEquiv {α} [Subsingleton α] : α × α ≃ α where
  toFun p := p.1
  invFun a := (a, a)
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


/-- To give an equivalence between two subsingleton types, it is sufficient to give any two
    functions between them. -/
def equivOfSubsingletonOfSubsingleton [Subsingleton α] [Subsingleton β] (f : α → β) (g : β → α) :
    α ≃ β where
  toFun := f
  invFun := g
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


/-- A nonempty subsingleton type is (noncomputably) equivalent to `PUnit`. -/
noncomputable def Equiv.punitOfNonemptyOfSubsingleton [h : Nonempty α] [Subsingleton α] :
    α ≃ PUnit :=
  equivOfSubsingletonOfSubsingleton (fun _ => PUnit.unit) fun _ => h.some


/-- `Unique (Unique α)` is equivalent to `Unique α`. -/
def uniqueUniqueEquiv : Unique (Unique α) ≃ Unique α :=
  equivOfSubsingletonOfSubsingleton (fun h => h.default) fun h =>
    { default := h, uniq := fun _ => Subsingleton.elim _ _ }


/-- If `Unique β`, then `Unique α` is equivalent to `α ≃ β`. -/
def uniqueEquivEquivUnique (α : Sort u) (β : Sort v) [Unique β] : Unique α ≃ (α ≃ β) :=
  equivOfSubsingletonOfSubsingleton (fun _ => Equiv.ofUnique _ _) Equiv.unique


theorem update_comp_equiv [DecidableEq α'] [DecidableEq α] (f : α → β)
    (g : α' ≃ α) (a : α) (v : β) :
    update f a v ∘ g = update (f ∘ g) (g.symm a) v := by
  /-
    α : Sort u_1
    β : Sort u_4
    α' : Sort u_9
    inst✝¹ : DecidableEq α'
    inst✝ : DecidableEq α
    f : α → β
    g : Equiv α' α
    a : α
    v : β
    ⊢ Eq (Function.comp (Function.update f a v) ⇑g) (Function.update (Function.com …
  -/
  rw [← update_comp_eq_of_injective _ g.injective, g.apply_symm_apply]
  /-
    🎉 no goals
  -/


theorem update_apply_equiv_apply [DecidableEq α'] [DecidableEq α] (f : α → β)
    (g : α' ≃ α) (a : α) (v : β) (a' : α') : update f a v (g a') = update (f ∘ g) (g.symm a) v a' :=
  congr_fun (update_comp_equiv f g a v) a'

-- Porting note: EmbeddingLike.apply_eq_iff_eq broken here too

theorem piCongrLeft'_update [DecidableEq α] [DecidableEq β] (P : α → Sort*) (e : α ≃ β)
    (f : ∀ a, P a) (b : β) (x : P (e.symm b)) :
    e.piCongrLeft' P (update f (e.symm b) x) = update (e.piCongrLeft' P f) b x := by
  /-
    α : Sort u_1
    β : Sort u_4
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    P : α → Sort u_10
    e : Equiv α β
    f : (a : α) → P a
    b : β
    x : P (e.symm b)
    ⊢ Eq ((Equiv.piCongrLeft' P e) (Function.update f (e.symm b) x)) (Function.upd …
  -/
  ext b'
  /-
    case h
    α : Sort u_1
    β : Sort u_4
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    P : α → Sort u_10
    e : Equiv α β
    f : (a : α) → P a
    b : β
    x : P (e.symm b)
    b' : β
    ⊢ Eq ((Equiv.piCongrLeft' P e) (Function.update f (e.symm b) x) b') (Function. …
  -/
  rcases eq_or_ne b' b with (rfl | h)
    /-
      case h.inl
      α : Sort u_1
      β : Sort u_4
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      P : α → Sort u_10
      e : Equiv α β
      f : (a : α) → P a
      b' : β
      x : P (e.symm b')
      ⊢ Eq ((Equiv.piCongrLeft' P e) (Function.update f (e.symm b') x) b') (Function …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      α : Sort u_1
      β : Sort u_4
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      P : α → Sort u_10
      e : Equiv α β
      f : (a : α) → P a
      b : β
      x : P (e.symm b)
      b' : β
      h : Ne b' b
      ⊢ Eq ((Equiv.piCongrLeft' P e) (Function.update f (e.symm b) x) b') (Function. …
    -/
  · simp only [Equiv.piCongrLeft'_apply, ne_eq, h, not_false_iff, update_of_ne]
    /-
      case h.inr
      α : Sort u_1
      β : Sort u_4
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      P : α → Sort u_10
      e : Equiv α β
      f : (a : α) → P a
      b : β
      x : P (e.symm b)
      b' : β
      h : Ne b' b
      ⊢ Eq (Function.update f (e.symm b) x (e.symm b')) (f (e.symm b'))
    -/
    rw [update_of_ne]
    /-
      case h.inr.h
      α : Sort u_1
      β : Sort u_4
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      P : α → Sort u_10
      e : Equiv α β
      f : (a : α) → P a
      b : β
      x : P (e.symm b)
      b' : β
      h : Ne b' b
      ⊢ Ne (e.symm b') (e.symm b)
    -/
    intro h'
    /- an example of something that should work, or also putting `EmbeddingLike.apply_eq_iff_eq`
      in the `simp` should too:
    have := (EmbeddingLike.apply_eq_iff_eq e).mp h' -/
    /-
      case h.inr.h
      α : Sort u_1
      β : Sort u_4
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      P : α → Sort u_10
      e : Equiv α β
      f : (a : α) → P a
      b : β
      x : P (e.symm b)
      b' : β
      h : Ne b' b
      h' : Eq (e.symm b') (e.symm b)
      ⊢ False
    -/
    cases e.symm.injective h' |> h
    /-
      🎉 no goals
    -/


theorem piCongrLeft'_symm_update [DecidableEq α] [DecidableEq β] (P : α → Sort*) (e : α ≃ β)
    (f : ∀ b, P (e.symm b)) (b : β) (x : P (e.symm b)) :
    (e.piCongrLeft' P).symm (update f b x) = update ((e.piCongrLeft' P).symm f) (e.symm b) x := by
  /-
    α : Sort u_1
    β : Sort u_4
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    P : α → Sort u_10
    e : Equiv α β
    f : (b : β) → P (e.symm b)
    b : β
    x : P (e.symm b)
    ⊢ Eq ((Equiv.piCongrLeft' P e).symm (Function.update f b x)) (Function.update  …
  -/
  simp [(e.piCongrLeft' P).symm_apply_eq, piCongrLeft'_update]
  /-
    🎉 no goals
  -/


