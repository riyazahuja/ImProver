theorem const_eventuallyEq' [NeBot l] {a b : β} : (∀ᶠ _ in l, a = b) ↔ a = b :=
  eventually_const


theorem const_eventuallyEq [NeBot l] {a b : β} : ((fun _ => a) =ᶠ[l] fun _ => b) ↔ a = b :=
  @const_eventuallyEq' _ _ _ _ a b


/-- Setoid used to define the space of germs. -/
def germSetoid (l : Filter α) (β : Type*) : Setoid (α → β) where
  r := EventuallyEq l
  iseqv := ⟨EventuallyEq.refl _, EventuallyEq.symm, EventuallyEq.trans⟩


/-- The space of germs of functions `α → β` at a filter `l`. -/
def Germ (l : Filter α) (β : Type*) : Type _ :=
  Quotient (germSetoid l β)


/-- Setoid used to define the filter product. This is a dependent version of
  `Filter.germSetoid`. -/
def productSetoid (l : Filter α) (ε : α → Type*) : Setoid ((a : _) → ε a) where
  r f g := ∀ᶠ a in l, f a = g a
  iseqv :=
    ⟨fun _ => Eventually.of_forall fun _ => rfl, fun h => h.mono fun _ => Eq.symm,
      fun h1 h2 => h1.congr (h2.mono fun _ hx => hx ▸ Iff.rfl)⟩


/-- The filter product `(a : α) → ε a` at a filter `l`. This is a dependent version of
  `Filter.Germ`. -/
def Product (l : Filter α) (ε : α → Type*) : Type _ :=
  Quotient (productSetoid l ε)


instance coeTC : CoeTC ((a : _) → ε a) (l.Product ε) :=
  ⟨@Quotient.mk' _ (productSetoid _ ε)⟩


instance instInhabited [(a : _) → Inhabited (ε a)] : Inhabited (l.Product ε) :=
  ⟨(↑fun a => (default : ε a) : l.Product ε)⟩


@[coe]
def ofFun : (α → β) → (Germ l β) := @Quotient.mk' _ (germSetoid _ _)


instance : CoeTC (α → β) (Germ l β) :=
  ⟨ofFun⟩


@[coe] -- Porting note: removed `HasLiftT` instance
def const {l : Filter α} (b : β) : (Germ l β) := ofFun fun _ => b


instance coeTC : CoeTC β (Germ l β) :=
  ⟨const⟩


/-- A germ `P` of functions `α → β` is constant w.r.t. `l`. -/
def IsConstant {l : Filter α} (P : Germ l β) : Prop :=
  P.liftOn (fun f ↦ ∃ b : β, f =ᶠ[l] (fun _ ↦ b)) <| by
    suffices ∀ f g : α → β, ∀ b : β, f =ᶠ[l] g → (f =ᶠ[l] fun _ ↦ b) → (g =ᶠ[l] fun _ ↦ b) from
      fun f g h ↦ propext ⟨fun ⟨b, hb⟩ ↦ ⟨b, this f g b h hb⟩, fun ⟨b, hb⟩ ↦ ⟨b, h.trans hb⟩⟩
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      l✝ : Filter α
      f g h : α → β
      l : Filter α
      P : l.Germ β
      ⊢ ∀ (f g : α → β) (b : β), l.EventuallyEq f g → (l.EventuallyEq f fun x => b)  …
    -/
    exact fun f g b hfg hf ↦ (hfg.symm).trans hf
    /-
      🎉 no goals
    -/


theorem isConstant_coe {l : Filter α} {b} (h : ∀ x', f x' = b) : (↑f : Germ l β).IsConstant :=
  ⟨b, Eventually.of_forall h⟩


@[simp]
theorem isConstant_coe_const {l : Filter α} {b : β} : (fun _ : α ↦ b : Germ l β).IsConstant := by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    b : β
    ⊢ (↑fun x => b).IsConstant
  -/
  use b
  /-
    🎉 no goals
  -/


/-- If `f : α → β` is constant w.r.t. `l` and `g : β → γ`, then `g ∘ f : α → γ` also is. -/
lemma isConstant_comp {l : Filter α} {f : α → β} {g : β → γ}
    (h : (f : Germ l β).IsConstant) : ((g ∘ f) : Germ l γ).IsConstant := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    l : Filter α
    f : α → β
    g : β → γ
    h : (↑f).IsConstant
    ⊢ (↑(Function.comp g f)).IsConstant
  -/
  obtain ⟨b, hb⟩ := h
  /-
    case intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    l : Filter α
    f : α → β
    g : β → γ
    b : β
    hb : l.EventuallyEq f fun x => b
    ⊢ (↑(Function.comp g f)).IsConstant
  -/
  exact ⟨g b, hb.fun_comp g⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem quot_mk_eq_coe (l : Filter α) (f : α → β) : Quot.mk _ f = (f : Germ l β) :=
  rfl


@[simp]
theorem mk'_eq_coe (l : Filter α) (f : α → β) :
    @Quotient.mk' _ (germSetoid _ _) f = (f : Germ l β) :=
  rfl


@[elab_as_elim]
theorem inductionOn (f : Germ l β) {p : Germ l β → Prop} (h : ∀ f : α → β, p f) : p f :=
  Quotient.inductionOn' f h


@[elab_as_elim]
theorem inductionOn₂ (f : Germ l β) (g : Germ l γ) {p : Germ l β → Germ l γ → Prop}
    (h : ∀ (f : α → β) (g : α → γ), p f g) : p f g :=
  Quotient.inductionOn₂' f g h


@[elab_as_elim]
theorem inductionOn₃ (f : Germ l β) (g : Germ l γ) (h : Germ l δ)
    {p : Germ l β → Germ l γ → Germ l δ → Prop}
    (H : ∀ (f : α → β) (g : α → γ) (h : α → δ), p f g h) : p f g h :=
  Quotient.inductionOn₃' f g h H


/-- Given a map `F : (α → β) → (γ → δ)` that sends functions eventually equal at `l` to functions
eventually equal at `lc`, returns a map from `Germ l β` to `Germ lc δ`. -/
def map' {lc : Filter γ} (F : (α → β) → γ → δ) (hF : (l.EventuallyEq ⇒ lc.EventuallyEq) F F) :
    Germ l β → Germ lc δ :=
  Quotient.map' F hF


/-- Given a germ `f : Germ l β` and a function `F : (α → β) → γ` sending eventually equal functions
to the same value, returns the value `F` takes on functions having germ `f` at `l`. -/
def liftOn {γ : Sort*} (f : Germ l β) (F : (α → β) → γ) (hF : (l.EventuallyEq ⇒ (· = ·)) F F) :
    γ :=
  Quotient.liftOn' f F hF


@[simp]
theorem map'_coe {lc : Filter γ} (F : (α → β) → γ → δ) (hF : (l.EventuallyEq ⇒ lc.EventuallyEq) F F)
    (f : α → β) : map' F hF f = F f :=
  rfl


@[simp, norm_cast]
theorem coe_eq : (f : Germ l β) = g ↔ f =ᶠ[l] g :=
  Quotient.eq''


alias ⟨_, _root_.Filter.EventuallyEq.germ_eq⟩ := coe_eq


/-- Lift a function `β → γ` to a function `Germ l β → Germ l γ`. -/
def map (op : β → γ) : Germ l β → Germ l γ :=
  map' (op ∘ ·) fun _ _ H => H.mono fun _ H => congr_arg op H


@[simp]
theorem map_coe (op : β → γ) (f : α → β) : map op (f : Germ l β) = op ∘ f :=
  rfl


@[simp]
theorem map_id : map id = (id : Germ l β → Germ l β) := by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    ⊢ Eq (Filter.Germ.map id) id
  -/
  ext ⟨f⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    l : Filter α
    x✝ : l.Germ β
    f : α → β
    ⊢ Eq (Filter.Germ.map id (Quot.mk (⇑(l.germSetoid β)) f)) (id (Quot.mk (⇑(l.ge …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem map_map (op₁ : γ → δ) (op₂ : β → γ) (f : Germ l β) :
    map op₁ (map op₂ f) = map (op₁ ∘ op₂) f :=
  inductionOn f fun _ => rfl


/-- Lift a binary function `β → γ → δ` to a function `Germ l β → Germ l γ → Germ l δ`. -/
def map₂ (op : β → γ → δ) : Germ l β → Germ l γ → Germ l δ :=
  Quotient.map₂ (fun f g x => op (f x) (g x)) fun f f' Hf g g' Hg =>
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         γ : Type u_3
                                         δ : Type u_4
                                         l : Filter α
                                         f✝ g✝ h : α → β
                                         op : β → γ → δ
                                         f f' : α → β
                                         Hf✝ : HasEquiv.Equiv f f'
                                         g g' : α → γ
                                         Hg✝ : HasEquiv.Equiv g g'
                                         x : α
                                         Hf : Eq (f x) (f' x)
                                         Hg : Eq (g x) (g' x)
                                         ⊢ Eq ((fun f g x => op (f x) (g x)) f g x) ((fun f g x => op (f x) (g x)) f' g …
                                       -/
    Hg.mp <| Hf.mono fun x Hf Hg => by simp only [Hf, Hg]
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem map₂_coe (op : β → γ → δ) (f : α → β) (g : α → γ) :
    map₂ op (f : Germ l β) g = fun x => op (f x) (g x) :=
  rfl


/-- A germ at `l` of maps from `α` to `β` tends to `lb : Filter β` if it is represented by a map
which tends to `lb` along `l`. -/
protected def Tendsto (f : Germ l β) (lb : Filter β) : Prop :=
  liftOn f (fun f => Tendsto f l lb) fun _f _g H => propext (tendsto_congr' H)


@[simp, norm_cast]
theorem coe_tendsto {f : α → β} {lb : Filter β} : (f : Germ l β).Tendsto lb ↔ Tendsto f l lb :=
  Iff.rfl


alias ⟨_, _root_.Filter.Tendsto.germ_tendsto⟩ := coe_tendsto


/-- Given two germs `f : Germ l β`, and `g : Germ lc α`, where `l : Filter α`, if `g` tends to `l`,
then the composition `f ∘ g` is well-defined as a germ at `lc`. -/
def compTendsto' (f : Germ l β) {lc : Filter γ} (g : Germ lc α) (hg : g.Tendsto l) : Germ lc β :=
  liftOn f (fun f => g.map f) fun _f₁ _f₂ hF =>
    inductionOn g (fun _g hg => coe_eq.2 <| hg.eventually hF) hg


@[simp]
theorem coe_compTendsto' (f : α → β) {lc : Filter γ} {g : Germ lc α} (hg : g.Tendsto l) :
    (f : Germ l β).compTendsto' g hg = g.map f :=
  rfl


/-- Given a germ `f : Germ l β` and a function `g : γ → α`, where `l : Filter α`, if `g` tends
to `l` along `lc : Filter γ`, then the composition `f ∘ g` is well-defined as a germ at `lc`. -/
def compTendsto (f : Germ l β) {lc : Filter γ} (g : γ → α) (hg : Tendsto g lc l) : Germ lc β :=
  f.compTendsto' _ hg.germ_tendsto


@[simp]
theorem coe_compTendsto (f : α → β) {lc : Filter γ} {g : γ → α} (hg : Tendsto g lc l) :
    (f : Germ l β).compTendsto g hg = f ∘ g :=
  rfl

-- Porting note https://github.com/leanprover-community/mathlib4/issues/10959
-- simp can't match the LHS.

@[simp, nolint simpNF]
theorem compTendsto'_coe (f : Germ l β) {lc : Filter γ} {g : γ → α} (hg : Tendsto g lc l) :
    f.compTendsto' _ hg.germ_tendsto = f.compTendsto g hg :=
  rfl


theorem Filter.Tendsto.congr_germ {f g : β → γ} {l : Filter α} {l' : Filter β} (h : f =ᶠ[l'] g)
    {φ : α → β} (hφ : Tendsto φ l l') : (f ∘ φ : Germ l γ) = g ∘ φ :=
  EventuallyEq.germ_eq (h.comp_tendsto hφ)


lemma isConstant_comp_tendsto {lc : Filter γ} {g : γ → α}
    (hf : (f : Germ l β).IsConstant) (hg : Tendsto g lc l) : IsConstant (f ∘ g : Germ lc β) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    l : Filter α
    f : α → β
    lc : Filter γ
    g : γ → α
    hf : (↑f).IsConstant
    hg : Filter.Tendsto g lc l
    ⊢ (↑(Function.comp f g)).IsConstant
  -/
  rcases hf with ⟨b, hb⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    l : Filter α
    f : α → β
    lc : Filter γ
    g : γ → α
    hg : Filter.Tendsto g lc l
    b : β
    hb : l.EventuallyEq f fun x => b
    ⊢ (↑(Function.comp f g)).IsConstant
  -/
  exact ⟨b, hb.comp_tendsto hg⟩
  /-
    🎉 no goals
  -/


/-- If a germ `f : Germ l β` is constant, where `l : Filter α`,
and a function `g : γ → α` tends to `l` along `lc : Filter γ`,
the germ of the composition `f ∘ g` is also constant. -/
lemma isConstant_compTendsto {f : Germ l β} {lc : Filter γ} {g : γ → α}
    (hf : f.IsConstant) (hg : Tendsto g lc l) : (f.compTendsto g hg).IsConstant := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    l : Filter α
    f : l.Germ β
    lc : Filter γ
    g : γ → α
    hf : f.IsConstant
    hg : Filter.Tendsto g lc l
    ⊢ (f.compTendsto g hg).IsConstant
  -/
  induction f using Quotient.inductionOn with | _ f => ?_
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    l : Filter α
    lc : Filter γ
    g : γ → α
    hg : Filter.Tendsto g lc l
    f : α → β
    hf : Filter.Germ.IsConstant (Quotient.mk (l.germSetoid β) f)
    ⊢ (Filter.Germ.compTendsto (Quotient.mk (l.germSetoid β) f) g hg).IsConstant
  -/
  exact isConstant_comp_tendsto hf hg
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem const_inj [NeBot l] {a b : β} : (↑a : Germ l β) = ↑b ↔ a = b :=
  coe_eq.trans const_eventuallyEq


@[simp]
theorem map_const (l : Filter α) (a : β) (f : β → γ) : (↑a : Germ l β).map f = ↑(f a) :=
  rfl


@[simp]
theorem map₂_const (l : Filter α) (b : β) (c : γ) (f : β → γ → δ) :
    map₂ f (↑b : Germ l β) ↑c = ↑(f b c) :=
  rfl


@[simp]
theorem const_compTendsto {l : Filter α} (b : β) {lc : Filter γ} {g : γ → α} (hg : Tendsto g lc l) :
    (↑b : Germ l β).compTendsto g hg = ↑b :=
  rfl


@[simp]
theorem const_compTendsto' {l : Filter α} (b : β) {lc : Filter γ} {g : Germ lc α}
    (hg : g.Tendsto l) : (↑b : Germ l β).compTendsto' g hg = ↑b :=
  inductionOn g (fun _ _ => rfl) hg


/-- Lift a predicate on `β` to `Germ l β`. -/
def LiftPred (p : β → Prop) (f : Germ l β) : Prop :=
  liftOn f (fun f => ∀ᶠ x in l, p (f x)) fun _f _g H =>
    propext <| eventually_congr <| H.mono fun _x hx => hx ▸ Iff.rfl


@[simp]
theorem liftPred_coe {p : β → Prop} {f : α → β} : LiftPred p (f : Germ l β) ↔ ∀ᶠ x in l, p (f x) :=
  Iff.rfl


theorem liftPred_const {p : β → Prop} {x : β} (hx : p x) : LiftPred p (↑x : Germ l β) :=
  Eventually.of_forall fun _y => hx


@[simp]
theorem liftPred_const_iff [NeBot l] {p : β → Prop} {x : β} : LiftPred p (↑x : Germ l β) ↔ p x :=
  @eventually_const _ _ _ (p x)


/-- Lift a relation `r : β → γ → Prop` to `Germ l β → Germ l γ → Prop`. -/
def LiftRel (r : β → γ → Prop) (f : Germ l β) (g : Germ l γ) : Prop :=
  Quotient.liftOn₂' f g (fun f g => ∀ᶠ x in l, r (f x) (g x)) fun _f _g _f' _g' Hf Hg =>
    propext <| eventually_congr <| Hg.mp <| Hf.mono fun _x hf hg => hf ▸ hg ▸ Iff.rfl


@[simp]
theorem liftRel_coe {r : β → γ → Prop} {f : α → β} {g : α → γ} :
    LiftRel r (f : Germ l β) g ↔ ∀ᶠ x in l, r (f x) (g x) :=
  Iff.rfl


theorem liftRel_const {r : β → γ → Prop} {x : β} {y : γ} (h : r x y) :
    LiftRel r (↑x : Germ l β) ↑y :=
  Eventually.of_forall fun _ => h


@[simp]
theorem liftRel_const_iff [NeBot l] {r : β → γ → Prop} {x : β} {y : γ} :
    LiftRel r (↑x : Germ l β) ↑y ↔ r x y :=
  @eventually_const _ _ _ (r x y)


instance instInhabited [Inhabited β] : Inhabited (Germ l β) := ⟨↑(default : β)⟩


@[to_additive] instance instMul [Mul M] : Mul (Germ l M) := ⟨map₂ (· * ·)⟩


@[to_additive (attr := simp, norm_cast)]
theorem coe_mul [Mul M] (f g : α → M) : ↑(f * g) = (f * g : Germ l M) :=
  rfl


@[to_additive] instance instOne [One M] : One (Germ l M) := ⟨↑(1 : M)⟩


@[to_additive (attr := simp, norm_cast)]
theorem coe_one [One M] : ↑(1 : α → M) = (1 : Germ l M) :=
  rfl


@[to_additive]
instance instSemigroup [Semigroup M] : Semigroup (Germ l M) :=
  { mul_assoc := fun a b c => Quotient.inductionOn₃' a b c
      fun _ _ _ => congrArg ofFun <| mul_assoc .. }


@[to_additive]
instance instCommSemigroup [CommSemigroup M] : CommSemigroup (Germ l M) :=
  { mul_comm := Quotient.ind₂' fun _ _ => congrArg ofFun <| mul_comm .. }


@[to_additive]
instance instIsLeftCancelMul [Mul M] [IsLeftCancelMul M] : IsLeftCancelMul (Germ l M) where
  mul_left_cancel f₁ f₂ f₃ :=
    inductionOn₃ f₁ f₂ f₃ fun _f₁ _f₂ _f₃ H =>
      coe_eq.2 ((coe_eq.1 H).mono fun _x => mul_left_cancel)


@[to_additive]
instance instIsRightCancelMul [Mul M] [IsRightCancelMul M] : IsRightCancelMul (Germ l M) where
  mul_right_cancel f₁ f₂ f₃ :=
    inductionOn₃ f₁ f₂ f₃ fun _f₁ _f₂ _f₃ H =>
      coe_eq.2 <| (coe_eq.1 H).mono fun _x => mul_right_cancel


@[to_additive]
instance instIsCancelMul [Mul M] [IsCancelMul M] : IsCancelMul (Germ l M) where


@[to_additive]
instance instLeftCancelSemigroup [LeftCancelSemigroup M] : LeftCancelSemigroup (Germ l M) where
  mul_left_cancel _ _ _ := mul_left_cancel


@[to_additive]
instance instRightCancelSemigroup [RightCancelSemigroup M] : RightCancelSemigroup (Germ l M) where
  mul_right_cancel _ _ _ := mul_right_cancel


@[to_additive]
instance instMulOneClass [MulOneClass M] : MulOneClass (Germ l M) :=
  { one_mul := Quotient.ind' fun _ => congrArg ofFun <| one_mul _
    mul_one := Quotient.ind' fun _ => congrArg ofFun <| mul_one _ }


@[to_additive]
instance instSMul [SMul M G] : SMul M (Germ l G) where smul n := map (n • ·)


@[to_additive existing instSMul]
instance instPow [Pow G M] : Pow (Germ l G) M where pow f n := map (· ^ n) f


@[to_additive (attr := simp, norm_cast)]
theorem coe_smul [SMul M G] (n : M) (f : α → G) : ↑(n • f) = n • (f : Germ l G) :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem const_smul [SMul M G] (n : M) (a : G) : (↑(n • a) : Germ l G) = n • (↑a : Germ l G) :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem coe_pow [Pow G M] (f : α → G) (n : M) : ↑(f ^ n) = (f : Germ l G) ^ n :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem const_pow [Pow G M] (a : G) (n : M) : (↑(a ^ n) : Germ l G) = (↑a : Germ l G) ^ n :=
  rfl

-- TODO: https://github.com/leanprover-community/mathlib4/pull/7432

@[to_additive]
instance instMonoid [Monoid M] : Monoid (Germ l M) :=
                                                            /-
                                                              α : Type u_1
                                                              β : Type u_2
                                                              γ : Type u_3
                                                              δ : Type u_4
                                                              l : Filter α
                                                              f g h : α → β
                                                              M : Type u_5
                                                              G : Type u_6
                                                              inst✝ : Monoid M
                                                              ⊢ Eq (↑1) 1
                                                            -/
  { Function.Surjective.monoid ofFun Quot.mk_surjective (by rfl)
                                                            /-
                                                              🎉 no goals
                                                            -/
                     /-
                       α : Type u_1
                       β : Type u_2
                       γ : Type u_3
                       δ : Type u_4
                       l : Filter α
                       f g h : α → β
                       M : Type u_5
                       G : Type u_6
                       inst✝ : Monoid M
                       x✝¹ x✝ : α → M
                       ⊢ Eq (↑(HMul.hMul x✝¹ x✝)) (HMul.hMul ↑x✝¹ ↑x✝)
                     -/
                     /-
                       🎉 no goals
                     -/
      (fun _ _ => by rfl) fun _ _ => by rfl with
                                        /-
                                          🎉 no goals
                                        -/
    toSemigroup := instSemigroup
    toOne := instOne
    npow := fun n a => a ^ n }


/-- Coercion from functions to germs as a monoid homomorphism. -/
@[to_additive "Coercion from functions to germs as an additive monoid homomorphism."]
def coeMulHom [Monoid M] (l : Filter α) : (α → M) →* Germ l M where
  toFun := ofFun; map_one' := rfl; map_mul' _ _ := rfl


@[to_additive (attr := simp)]
theorem coe_coeMulHom [Monoid M] : (coeMulHom l : (α → M) → Germ l M) = ofFun :=
  rfl


@[to_additive]
instance instCommMonoid [CommMonoid M] : CommMonoid (Germ l M) :=
  { mul_comm := mul_comm }


instance instNatCast [NatCast M] : NatCast (Germ l M) where natCast n := (n : α → M)


@[simp]
theorem natCast_def [NatCast M] (n : ℕ) : ((fun _ ↦ n : α → M) : Germ l M) = n := rfl


@[simp, norm_cast]
theorem const_nat [NatCast M] (n : ℕ) : ((n : M) : Germ l M) = n := rfl

-- See note [no_index around OfNat.ofNat]

@[simp, norm_cast]
theorem coe_ofNat [NatCast M] (n : ℕ) [n.AtLeastTwo] :
    ((no_index (OfNat.ofNat n : α → M)) : Germ l M) = OfNat.ofNat n :=
  rfl

-- See note [no_index around OfNat.ofNat]

@[simp, norm_cast]
theorem const_ofNat [NatCast M] (n : ℕ) [n.AtLeastTwo] :
    ((no_index (OfNat.ofNat n : M)) : Germ l M) = OfNat.ofNat n :=
  rfl


instance instIntCast [IntCast M] : IntCast (Germ l M) where intCast n := (n : α → M)


@[simp]
theorem intCast_def [IntCast M] (n : ℤ) : ((fun _ ↦ n : α → M) : Germ l M) = n := rfl


@[deprecated (since := "2024-04-05")] alias coe_nat := natCast_def

@[deprecated (since := "2024-04-05")] alias coe_int := intCast_def


instance instAddMonoidWithOne [AddMonoidWithOne M] : AddMonoidWithOne (Germ l M) where
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         γ : Type u_3
                                         δ : Type u_4
                                         l : Filter α
                                         f g h : α → β
                                         M : Type u_5
                                         G : Type u_6
                                         inst✝ : AddMonoidWithOne M
                                         ⊢ Eq ↑0 fun x => 0
                                       -/
  natCast_zero := congrArg ofFun <| by simp; rfl
                                             /-
                                               🎉 no goals
                                             -/
                                         /-
                                           α : Type u_1
                                           β : Type u_2
                                           γ : Type u_3
                                           δ : Type u_4
                                           l : Filter α
                                           f g h : α → β
                                           M : Type u_5
                                           G : Type u_6
                                           inst✝ : AddMonoidWithOne M
                                           x✝ : Nat
                                           ⊢ Eq (↑(HAdd.hAdd x✝ 1)) ((fun f g x => (fun x1 x2 => HAdd.hAdd x1 x2) (f x) ( …
                                         -/
  natCast_succ _ := congrArg ofFun <| by simp [Function.comp]; rfl
                                                               /-
                                                                 🎉 no goals
                                                               -/


instance instAddCommMonoidWithOne [AddCommMonoidWithOne M] : AddCommMonoidWithOne (Germ l M) :=
  { add_comm := add_comm }


@[to_additive] instance instInv [Inv G] : Inv (Germ l G) := ⟨map Inv.inv⟩


@[to_additive (attr := simp, norm_cast)]
theorem coe_inv [Inv G] (f : α → G) : ↑f⁻¹ = (f⁻¹ : Germ l G) :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem const_inv [Inv G] (a : G) : (↑(a⁻¹) : Germ l G) = (↑a)⁻¹ :=
  rfl


@[to_additive] instance instDiv [Div M] : Div (Germ l M) := ⟨map₂ (· / ·)⟩


@[to_additive (attr := simp, norm_cast)]
theorem coe_div [Div M] (f g : α → M) : ↑(f / g) = (f / g : Germ l M) :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem const_div [Div M] (a b : M) : (↑(a / b) : Germ l M) = ↑a / ↑b :=
  rfl


@[to_additive]
instance instInvolutiveInv [InvolutiveInv G] : InvolutiveInv (Germ l G) :=
  { inv_inv := Quotient.ind' fun _ => congrArg ofFun<| inv_inv _ }


instance instHasDistribNeg [Mul G] [HasDistribNeg G] : HasDistribNeg (Germ l G) :=
  { neg_mul := Quotient.ind₂' fun _ _ => congrArg ofFun <| neg_mul ..
    mul_neg := Quotient.ind₂' fun _ _ => congrArg ofFun <| mul_neg .. }


@[to_additive]
instance instInvOneClass [InvOneClass G] : InvOneClass (Germ l G) :=
  ⟨congr_arg ofFun inv_one⟩


@[to_additive subNegMonoid]
instance instDivInvMonoid [DivInvMonoid G] : DivInvMonoid (Germ l G) where
  zpow z f := f ^ z
  zpow_zero' := Quotient.ind' fun _ => congrArg ofFun <|
    funext fun _ => DivInvMonoid.zpow_zero' _
  zpow_succ' _ := Quotient.ind' fun _ => congrArg ofFun <|
    funext fun _ => DivInvMonoid.zpow_succ' ..
  zpow_neg' _ := Quotient.ind' fun _ => congrArg ofFun <|
    funext fun _ => DivInvMonoid.zpow_neg' ..
  div_eq_mul_inv := Quotient.ind₂' fun _ _ ↦ congrArg ofFun <| div_eq_mul_inv ..


@[to_additive]
instance instDivisionMonoid [DivisionMonoid G] : DivisionMonoid (Germ l G) where
  inv_inv := inv_inv
  mul_inv_rev x y := inductionOn₂ x y fun _ _ ↦ congr_arg ofFun <| mul_inv_rev _ _
  inv_eq_of_mul x y := inductionOn₂ x y fun _ _ h ↦ coe_eq.2 <| (coe_eq.1 h).mono fun _ ↦
    DivisionMonoid.inv_eq_of_mul _ _


@[to_additive]
instance instGroup [Group G] : Group (Germ l G) :=
  { inv_mul_cancel := Quotient.ind' fun _ => congrArg ofFun <| inv_mul_cancel _ }


@[to_additive]
instance instCommGroup [CommGroup G] : CommGroup (Germ l G) :=
  { mul_comm := mul_comm }


instance instAddGroupWithOne [AddGroupWithOne G] : AddGroupWithOne (Germ l G) where
  __ := instAddMonoidWithOne
  __ := instAddGroup
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            γ : Type u_3
                                            δ : Type u_4
                                            l : Filter α
                                            f g h : α → β
                                            M : Type u_5
                                            G : Type u_6
                                            inst✝ : AddGroupWithOne G
                                            x✝ : Nat
                                            ⊢ Eq ↑↑x✝ ↑x✝
                                          -/
  intCast_ofNat _ := congrArg ofFun <| by simp
                                          /-
                                            🎉 no goals
                                          -/
                                            /-
                                              α : Type u_1
                                              β : Type u_2
                                              γ : Type u_3
                                              δ : Type u_4
                                              l : Filter α
                                              f g h : α → β
                                              M : Type u_5
                                              G : Type u_6
                                              inst✝ : AddGroupWithOne G
                                              x✝ : Nat
                                              ⊢ Eq (↑(Int.negSucc x✝)) ((fun x => Function.comp Neg.neg x) ↑(HAdd.hAdd x✝ 1))
                                            -/
  intCast_negSucc _ := congrArg ofFun <| by simp [Function.comp_def]; rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


instance instNontrivial [Nontrivial R] [NeBot l] : Nontrivial (Germ l R) :=
  let ⟨x, y, h⟩ := exists_pair_ne R
  ⟨⟨↑x, ↑y, mt const_inj.1 h⟩⟩


instance instMulZeroClass [MulZeroClass R] : MulZeroClass (Germ l R) :=
  { zero_mul := Quotient.ind' fun _ => congrArg ofFun <| zero_mul _
    mul_zero := Quotient.ind' fun _ => congrArg ofFun <| mul_zero _ }


instance instMulZeroOneClass [MulZeroOneClass R] : MulZeroOneClass (Germ l R) where
  __ := instMulZeroClass
  __ := instMulOneClass


instance instMonoidWithZero [MonoidWithZero R] : MonoidWithZero (Germ l R) where
  __ := instMonoid
  __ := instMulZeroClass


instance instDistrib [Distrib R] : Distrib (Germ l R) where
  left_distrib a b c := Quotient.inductionOn₃' a b c fun _ _ _ ↦ congrArg ofFun <| left_distrib ..
  right_distrib a b c := Quotient.inductionOn₃' a b c fun _ _ _ ↦ congrArg ofFun <| right_distrib ..


instance instNonUnitalNonAssocSemiring [NonUnitalNonAssocSemiring R] :
    NonUnitalNonAssocSemiring (Germ l R) where
  __ := instAddCommMonoid
  __ := instDistrib
  __ := instMulZeroClass


instance instNonUnitalSemiring [NonUnitalSemiring R] : NonUnitalSemiring (Germ l R) :=
  { mul_assoc := mul_assoc }


instance instNonAssocSemiring [NonAssocSemiring R] : NonAssocSemiring (Germ l R) where
  __ := instNonUnitalNonAssocSemiring
  __ := instMulZeroOneClass
  __ := instAddMonoidWithOne


instance instNonUnitalNonAssocRing [NonUnitalNonAssocRing R] :
    NonUnitalNonAssocRing (Germ l R) where
  __ := instAddCommGroup
  __ := instNonUnitalNonAssocSemiring


instance instNonUnitalRing [NonUnitalRing R] : NonUnitalRing (Germ l R) :=
  { mul_assoc := mul_assoc }


instance instNonAssocRing [NonAssocRing R] : NonAssocRing (Germ l R) where
  __ := instNonUnitalNonAssocRing
  __ := instNonAssocSemiring
  __ := instAddGroupWithOne


instance instSemiring [Semiring R] : Semiring (Germ l R) where
  __ := instNonUnitalSemiring
  __ := instNonAssocSemiring
  __ := instMonoidWithZero


instance instRing [Ring R] : Ring (Germ l R) where
  __ := instSemiring
  __ := instAddCommGroup
  __ := instNonAssocRing


instance instNonUnitalCommSemiring [NonUnitalCommSemiring R] :
    NonUnitalCommSemiring (Germ l R) :=
  { mul_comm := mul_comm }


instance instCommSemiring [CommSemiring R] : CommSemiring (Germ l R) :=
  { mul_comm := mul_comm }


instance instNonUnitalCommRing [NonUnitalCommRing R] : NonUnitalCommRing (Germ l R) where
  __ := instNonUnitalRing
  __ := instCommSemigroup


instance instCommRing [CommRing R] : CommRing (Germ l R) :=
  { mul_comm := mul_comm }


/-- Coercion `(α → R) → Germ l R` as a `RingHom`. -/
def coeRingHom [Semiring R] (l : Filter α) : (α → R) →+* Germ l R :=
  { (coeMulHom l : _ →* Germ l R), (coeAddHom l : _ →+ Germ l R) with toFun := ofFun }


@[simp]
theorem coe_coeRingHom [Semiring R] : (coeRingHom l : (α → R) → Germ l R) = ofFun :=
  rfl


@[to_additive]
instance instSMul' [SMul M β] : SMul (Germ l M) (Germ l β) :=
  ⟨map₂ (· • ·)⟩


@[to_additive (attr := simp, norm_cast)]
theorem coe_smul' [SMul M β] (c : α → M) (f : α → β) : ↑(c • f) = (c : Germ l M) • (f : Germ l β) :=
  rfl


@[to_additive]
instance instMulAction [Monoid M] [MulAction M β] : MulAction M (Germ l β) where
  one_smul f :=
    inductionOn f fun f => by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        l : Filter α
        f✝¹ g h : α → β
        M : Type u_5
        N : Type u_6
        R : Type u_7
        inst✝¹ : Monoid M
        inst✝ : MulAction M β
        f✝ : l.Germ β
        f : α → β
        ⊢ Eq (HSMul.hSMul 1 ↑f) ↑f
      -/
      norm_cast
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        l : Filter α
        f✝¹ g h : α → β
        M : Type u_5
        N : Type u_6
        R : Type u_7
        inst✝¹ : Monoid M
        inst✝ : MulAction M β
        f✝ : l.Germ β
        f : α → β
        ⊢ l.EventuallyEq (HSMul.hSMul 1 f) f
      -/
      simp [one_smul]
      /-
        🎉 no goals
      -/
  mul_smul c₁ c₂ f :=
    inductionOn f fun f => by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        l : Filter α
        f✝¹ g h : α → β
        M : Type u_5
        N : Type u_6
        R : Type u_7
        inst✝¹ : Monoid M
        inst✝ : MulAction M β
        c₁ c₂ : M
        f✝ : l.Germ β
        f : α → β
        ⊢ Eq (HSMul.hSMul (HMul.hMul c₁ c₂) ↑f) (HSMul.hSMul c₁ (HSMul.hSMul c₂ ↑f))
      -/
      norm_cast
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        l : Filter α
        f✝¹ g h : α → β
        M : Type u_5
        N : Type u_6
        R : Type u_7
        inst✝¹ : Monoid M
        inst✝ : MulAction M β
        c₁ c₂ : M
        f✝ : l.Germ β
        f : α → β
        ⊢ l.EventuallyEq (HSMul.hSMul (HMul.hMul c₁ c₂) f) (HSMul.hSMul c₁ (HSMul.hSMu …
      -/
      simp [mul_smul]
      /-
        🎉 no goals
      -/


@[to_additive]
instance instMulAction' [Monoid M] [MulAction M β] : MulAction (Germ l M) (Germ l β) where
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            γ : Type u_3
                                            δ : Type u_4
                                            l : Filter α
                                            f✝¹ g h : α → β
                                            M : Type u_5
                                            N : Type u_6
                                            R : Type u_7
                                            inst✝¹ : Monoid M
                                            inst✝ : MulAction M β
                                            f✝ : l.Germ β
                                            f : α → β
                                            ⊢ Eq (HSMul.hSMul 1 ↑f) ↑f
                                          -/
  one_smul f := inductionOn f fun f => by simp only [← coe_one, ← coe_smul', one_smul]
                                          /-
                                            🎉 no goals
                                          -/
  mul_smul c₁ c₂ f :=
    inductionOn₃ c₁ c₂ f fun c₁ c₂ f => by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        l : Filter α
        f✝¹ g h : α → β
        M : Type u_5
        N : Type u_6
        R : Type u_7
        inst✝¹ : Monoid M
        inst✝ : MulAction M β
        c₁✝ c₂✝ : l.Germ M
        f✝ : l.Germ β
        c₁ c₂ : α → M
        f : α → β
        ⊢ Eq (HSMul.hSMul (HMul.hMul ↑c₁ ↑c₂) ↑f) (HSMul.hSMul (↑c₁) (HSMul.hSMul ↑c₂  …
      -/
      norm_cast
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        l : Filter α
        f✝¹ g h : α → β
        M : Type u_5
        N : Type u_6
        R : Type u_7
        inst✝¹ : Monoid M
        inst✝ : MulAction M β
        c₁✝ c₂✝ : l.Germ M
        f✝ : l.Germ β
        c₁ c₂ : α → M
        f : α → β
        ⊢ l.EventuallyEq (HSMul.hSMul (HMul.hMul c₁ c₂) f) (HSMul.hSMul c₁ (HSMul.hSMu …
      -/
      simp [mul_smul]
      /-
        🎉 no goals
      -/


instance instDistribMulAction [Monoid M] [AddMonoid N] [DistribMulAction M N] :
    DistribMulAction M (Germ l N) where
  smul_add c f g :=
    inductionOn₂ f g fun f g => by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        l : Filter α
        f✝¹ g✝¹ h : α → β
        M : Type u_5
        N : Type u_6
        R : Type u_7
        inst✝² : Monoid M
        inst✝¹ : AddMonoid N
        inst✝ : DistribMulAction M N
        c : M
        f✝ g✝ : l.Germ N
        f g : α → N
        ⊢ Eq (HSMul.hSMul c (HAdd.hAdd ↑f ↑g)) (HAdd.hAdd (HSMul.hSMul c ↑f) (HSMul.hS …
      -/
      norm_cast
                    /-
                      α : Type u_1
                      β : Type u_2
                      γ : Type u_3
                      δ : Type u_4
                      l : Filter α
                      f g h : α → β
                      M : Type u_5
                      N : Type u_6
                      R : Type u_7
                      inst✝² : Monoid M
                      inst✝¹ : AddMonoid N
                      inst✝ : DistribMulAction M N
                      c : M
                      ⊢ Eq (HSMul.hSMul c 0) 0
                    -/
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        l : Filter α
        f✝¹ g✝¹ h : α → β
        M : Type u_5
        N : Type u_6
        R : Type u_7
        inst✝² : Monoid M
        inst✝¹ : AddMonoid N
        inst✝ : DistribMulAction M N
        c : M
        f✝ g✝ : l.Germ N
        f g : α → N
        ⊢ l.EventuallyEq (HSMul.hSMul c (HAdd.hAdd f g)) (HAdd.hAdd (HSMul.hSMul c f)  …
      -/
                    /-
                      🎉 no goals
                    -/
      simp [smul_add]
      /-
        🎉 no goals
      -/
  smul_zero c := by simp only [← coe_zero, ← coe_smul, smul_zero]


instance instDistribMulAction' [Monoid M] [AddMonoid N] [DistribMulAction M N] :
    DistribMulAction (Germ l M) (Germ l N) where
  smul_add c f g :=
    inductionOn₃ c f g fun c f g => by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        l : Filter α
        f✝¹ g✝¹ h : α → β
        M : Type u_5
        N : Type u_6
        R : Type u_7
        inst✝² : Monoid M
        inst✝¹ : AddMonoid N
        inst✝ : DistribMulAction M N
        c✝ : l.Germ M
        f✝ g✝ : l.Germ N
        c : α → M
        f g : α → N
        ⊢ Eq (HSMul.hSMul (↑c) (HAdd.hAdd ↑f ↑g)) (HAdd.hAdd (HSMul.hSMul ↑c ↑f) (HSMu …
      -/
      norm_cast
                                           /-
                                             α : Type u_1
                                             β : Type u_2
                                             γ : Type u_3
                                             δ : Type u_4
                                             l : Filter α
                                             f g h : α → β
                                             M : Type u_5
                                             N : Type u_6
                                             R : Type u_7
                                             inst✝² : Monoid M
                                             inst✝¹ : AddMonoid N
                                             inst✝ : DistribMulAction M N
                                             c✝ : l.Germ M
                                             c : α → M
                                             ⊢ Eq (HSMul.hSMul (↑c) 0) 0
                                           -/
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        l : Filter α
        f✝¹ g✝¹ h : α → β
        M : Type u_5
        N : Type u_6
        R : Type u_7
        inst✝² : Monoid M
        inst✝¹ : AddMonoid N
        inst✝ : DistribMulAction M N
        c✝ : l.Germ M
        f✝ g✝ : l.Germ N
        c : α → M
        f g : α → N
        ⊢ l.EventuallyEq (HSMul.hSMul c (HAdd.hAdd f g)) (HAdd.hAdd (HSMul.hSMul c f)  …
      -/
                                           /-
                                             🎉 no goals
                                           -/
      simp [smul_add]
      /-
        🎉 no goals
      -/
  smul_zero c := inductionOn c fun c => by simp only [← coe_zero, ← coe_smul', smul_zero]


instance instModule [Semiring R] [AddCommMonoid M] [Module R M] : Module R (Germ l M) where
  add_smul c₁ c₂ f :=
    inductionOn f fun f => by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        l : Filter α
        f✝¹ g h : α → β
        M : Type u_5
        N : Type u_6
        R : Type u_7
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        c₁ c₂ : R
        f✝ : l.Germ M
        f : α → M
        ⊢ Eq (HSMul.hSMul (HAdd.hAdd c₁ c₂) ↑f) (HAdd.hAdd (HSMul.hSMul c₁ ↑f) (HSMul. …
      -/
      norm_cast
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        l : Filter α
        f✝¹ g h : α → β
        M : Type u_5
        N : Type u_6
        R : Type u_7
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        c₁ c₂ : R
        f✝ : l.Germ M
        f : α → M
        ⊢ l.EventuallyEq (HSMul.hSMul (HAdd.hAdd c₁ c₂) f) (HAdd.hAdd (HSMul.hSMul c₁  …
      -/
      simp [add_smul]
      /-
        🎉 no goals
      -/
  zero_smul f :=
    inductionOn f fun f => by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        l : Filter α
        f✝¹ g h : α → β
        M : Type u_5
        N : Type u_6
        R : Type u_7
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        f✝ : l.Germ M
        f : α → M
        ⊢ Eq (HSMul.hSMul 0 ↑f) 0
      -/
      norm_cast
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        l : Filter α
        f✝¹ g h : α → β
        M : Type u_5
        N : Type u_6
        R : Type u_7
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        f✝ : l.Germ M
        f : α → M
        ⊢ l.EventuallyEq (HSMul.hSMul 0 f) 0
      -/
      simp [zero_smul, coe_zero]
      /-
        🎉 no goals
      -/


instance instModule' [Semiring R] [AddCommMonoid M] [Module R M] :
    Module (Germ l R) (Germ l M) where
  add_smul c₁ c₂ f :=
    inductionOn₃ c₁ c₂ f fun c₁ c₂ f => by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        l : Filter α
        f✝¹ g h : α → β
        M : Type u_5
        N : Type u_6
        R : Type u_7
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        c₁✝ c₂✝ : l.Germ R
        f✝ : l.Germ M
        c₁ c₂ : α → R
        f : α → M
        ⊢ Eq (HSMul.hSMul (HAdd.hAdd ↑c₁ ↑c₂) ↑f) (HAdd.hAdd (HSMul.hSMul ↑c₁ ↑f) (HSM …
      -/
      norm_cast
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        l : Filter α
        f✝¹ g h : α → β
        M : Type u_5
        N : Type u_6
        R : Type u_7
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        c₁✝ c₂✝ : l.Germ R
        f✝ : l.Germ M
        c₁ c₂ : α → R
        f : α → M
        ⊢ l.EventuallyEq (HSMul.hSMul (HAdd.hAdd c₁ c₂) f) (HAdd.hAdd (HSMul.hSMul c₁  …
      -/
      simp [add_smul]
      /-
        🎉 no goals
      -/
                                           /-
                                             α : Type u_1
                                             β : Type u_2
                                             γ : Type u_3
                                             δ : Type u_4
                                             l : Filter α
                                             f✝¹ g h : α → β
                                             M : Type u_5
                                             N : Type u_6
                                             R : Type u_7
                                             inst✝² : Semiring R
                                             inst✝¹ : AddCommMonoid M
                                             inst✝ : Module R M
                                             f✝ : l.Germ M
                                             f : α → M
                                             ⊢ Eq (HSMul.hSMul 0 ↑f) 0
                                           -/
  zero_smul f := inductionOn f fun f => by simp only [← coe_zero, ← coe_smul', zero_smul]
                                           /-
                                             🎉 no goals
                                           -/


instance instLE [LE β] : LE (Germ l β) := ⟨LiftRel (· ≤ ·)⟩


theorem le_def [LE β] : ((· ≤ ·) : Germ l β → Germ l β → Prop) = LiftRel (· ≤ ·) :=
  rfl


@[simp]
theorem coe_le [LE β] : (f : Germ l β) ≤ g ↔ f ≤ᶠ[l] g :=
  Iff.rfl


theorem coe_nonneg [LE β] [Zero β] {f : α → β} : 0 ≤ (f : Germ l β) ↔ ∀ᶠ x in l, 0 ≤ f x :=
  Iff.rfl


theorem const_le [LE β] {x y : β} : x ≤ y → (↑x : Germ l β) ≤ ↑y :=
  liftRel_const


@[simp, norm_cast]
theorem const_le_iff [LE β] [NeBot l] {x y : β} : (↑x : Germ l β) ≤ ↑y ↔ x ≤ y :=
  liftRel_const_iff


instance instPreorder [Preorder β] : Preorder (Germ l β) where
  le := (· ≤ ·)
  le_refl f := inductionOn f <| EventuallyLE.refl l
  le_trans f₁ f₂ f₃ := inductionOn₃ f₁ f₂ f₃ fun _ _ _ => EventuallyLE.trans


instance instPartialOrder [PartialOrder β] : PartialOrder (Germ l β) where
  le_antisymm f g := inductionOn₂ f g fun _ _ h₁ h₂ ↦ (EventuallyLE.antisymm h₁ h₂).germ_eq


instance instBot [Bot β] : Bot (Germ l β) := ⟨↑(⊥ : β)⟩

instance instTop [Top β] : Top (Germ l β) := ⟨↑(⊤ : β)⟩


@[simp, norm_cast]
theorem const_bot [Bot β] : (↑(⊥ : β) : Germ l β) = ⊥ :=
  rfl


@[simp, norm_cast]
theorem const_top [Top β] : (↑(⊤ : β) : Germ l β) = ⊤ :=
  rfl


instance instOrderBot [LE β] [OrderBot β] : OrderBot (Germ l β) where
  bot_le f := inductionOn f fun _ => Eventually.of_forall fun _ => bot_le


instance instOrderTop [LE β] [OrderTop β] : OrderTop (Germ l β) where
  le_top f := inductionOn f fun _ => Eventually.of_forall fun _ => le_top


instance instBoundedOrder [LE β] [BoundedOrder β] : BoundedOrder (Germ l β) where
  __ := instOrderBot
  __ := instOrderTop


instance instSup [Max β] : Max (Germ l β) := ⟨map₂ (· ⊔ ·)⟩

instance instInf [Min β] : Min (Germ l β) := ⟨map₂ (· ⊓ ·)⟩


@[simp, norm_cast]
theorem const_sup [Max β] (a b : β) : ↑(a ⊔ b) = (↑a ⊔ ↑b : Germ l β) :=
  rfl


@[simp, norm_cast]
theorem const_inf [Min β] (a b : β) : ↑(a ⊓ b) = (↑a ⊓ ↑b : Germ l β) :=
  rfl


instance instSemilatticeSup [SemilatticeSup β] : SemilatticeSup (Germ l β) where
  sup := max
  le_sup_left f g := inductionOn₂ f g fun _f _g => Eventually.of_forall fun _x ↦ le_sup_left
  le_sup_right f g := inductionOn₂ f g fun _f _g ↦ Eventually.of_forall fun _x ↦ le_sup_right
  sup_le f₁ f₂ g := inductionOn₃ f₁ f₂ g fun _f₁ _f₂ _g h₁ h₂ ↦ h₂.mp <| h₁.mono fun _x ↦ sup_le


instance instSemilatticeInf [SemilatticeInf β] : SemilatticeInf (Germ l β) where
  inf := min
  inf_le_left f g := inductionOn₂ f g fun _f _g ↦ Eventually.of_forall fun _x ↦ inf_le_left
  inf_le_right f g := inductionOn₂ f g fun _f _g ↦ Eventually.of_forall fun _x ↦ inf_le_right
  le_inf f₁ f₂ g := inductionOn₃ f₁ f₂ g fun _f₁ _f₂ _g h₁ h₂ ↦ h₂.mp <| h₁.mono fun _x ↦ le_inf


instance instLattice [Lattice β] : Lattice (Germ l β) where
  __ := instSemilatticeSup
  __ := instSemilatticeInf


instance instDistribLattice [DistribLattice β] : DistribLattice (Germ l β) where
  le_sup_inf f g h := inductionOn₃ f g h fun _f _g _h ↦ Eventually.of_forall fun _ ↦ le_sup_inf


@[to_additive]
instance instExistsMulOfLE [Mul β] [LE β] [ExistsMulOfLE β] : ExistsMulOfLE (Germ l β) where
  exists_mul_of_le {x y} := inductionOn₂ x y fun f g (h : f ≤ᶠ[l] g) ↦ by
    classical
    choose c hc using fun x (hx : f x ≤ g x) ↦ exists_mul_of_le hx
    refine ⟨ofFun fun x ↦ if hx : f x ≤ g x then c x hx else f x, coe_eq.2 ?_⟩
    filter_upwards [h] with x hx
    rw [dif_pos hx, hc]


