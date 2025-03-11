@[to_additive]
instance instMul [Mul M] [Mul N] : Mul (M × N) :=
  ⟨fun p q => ⟨p.1 * q.1, p.2 * q.2⟩⟩


@[to_additive (attr := simp)]
theorem fst_mul [Mul M] [Mul N] (p q : M × N) : (p * q).1 = p.1 * q.1 :=
  rfl


@[to_additive (attr := simp)]
theorem snd_mul [Mul M] [Mul N] (p q : M × N) : (p * q).2 = p.2 * q.2 :=
  rfl


@[to_additive (attr := simp)]
theorem mk_mul_mk [Mul M] [Mul N] (a₁ a₂ : M) (b₁ b₂ : N) :
    (a₁, b₁) * (a₂, b₂) = (a₁ * a₂, b₁ * b₂) :=
  rfl


@[to_additive (attr := simp)]
theorem swap_mul [Mul M] [Mul N] (p q : M × N) : (p * q).swap = p.swap * q.swap :=
  rfl


@[to_additive]
theorem mul_def [Mul M] [Mul N] (p q : M × N) : p * q = (p.1 * q.1, p.2 * q.2) :=
  rfl


@[to_additive]
theorem one_mk_mul_one_mk [Monoid M] [Mul N] (b₁ b₂ : N) :
    ((1 : M), b₁) * (1, b₂) = (1, b₁ * b₂) := by
  /-
    M : Type u_3
    N : Type u_4
    inst✝¹ : Monoid M
    inst✝ : Mul N
    b₁ b₂ : N
    ⊢ Eq (HMul.hMul { fst := 1, snd := b₁ } { fst := 1, snd := b₂ }) { fst := 1, s …
  -/
  rw [mk_mul_mk, mul_one]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mk_one_mul_mk_one [Mul M] [Monoid N] (a₁ a₂ : M) :
    (a₁, (1 : N)) * (a₂, 1) = (a₁ * a₂, 1) := by
  /-
    M : Type u_3
    N : Type u_4
    inst✝¹ : Mul M
    inst✝ : Monoid N
    a₁ a₂ : M
    ⊢ Eq (HMul.hMul { fst := a₁, snd := 1 } { fst := a₂, snd := 1 }) { fst := HMul …
  -/
  rw [mk_mul_mk, mul_one]
  /-
    🎉 no goals
  -/


@[to_additive]
instance instOne [One M] [One N] : One (M × N) :=
  ⟨(1, 1)⟩


@[to_additive (attr := simp)]
theorem fst_one [One M] [One N] : (1 : M × N).1 = 1 :=
  rfl


@[to_additive (attr := simp)]
theorem snd_one [One M] [One N] : (1 : M × N).2 = 1 :=
  rfl


@[to_additive]
theorem one_eq_mk [One M] [One N] : (1 : M × N) = (1, 1) :=
  rfl


@[to_additive (attr := simp)]
theorem mk_one_one [One M] [One N] : ((1 : M), (1 : N)) = 1 := rfl


@[to_additive (attr := simp)]
theorem mk_eq_one [One M] [One N] {x : M} {y : N} : (x, y) = 1 ↔ x = 1 ∧ y = 1 :=
  mk.inj_iff


@[to_additive (attr := simp)]
theorem swap_one [One M] [One N] : (1 : M × N).swap = 1 :=
  rfl


@[to_additive]
theorem fst_mul_snd [MulOneClass M] [MulOneClass N] (p : M × N) : (p.fst, 1) * (1, p.snd) = p :=
  Prod.ext (mul_one p.1) (one_mul p.2)


@[to_additive]
instance instInv [Inv M] [Inv N] : Inv (M × N) :=
  ⟨fun p => (p.1⁻¹, p.2⁻¹)⟩


@[to_additive (attr := simp)]
theorem fst_inv [Inv G] [Inv H] (p : G × H) : p⁻¹.1 = p.1⁻¹ :=
  rfl


@[to_additive (attr := simp)]
theorem snd_inv [Inv G] [Inv H] (p : G × H) : p⁻¹.2 = p.2⁻¹ :=
  rfl


@[to_additive (attr := simp)]
theorem inv_mk [Inv G] [Inv H] (a : G) (b : H) : (a, b)⁻¹ = (a⁻¹, b⁻¹) :=
  rfl


@[to_additive (attr := simp)]
theorem swap_inv [Inv G] [Inv H] (p : G × H) : p⁻¹.swap = p.swap⁻¹ :=
  rfl


@[to_additive]
instance [InvolutiveInv M] [InvolutiveInv N] : InvolutiveInv (M × N) :=
  { inv_inv := fun _ => Prod.ext (inv_inv _) (inv_inv _) }


@[to_additive]
instance instDiv [Div M] [Div N] : Div (M × N) :=
  ⟨fun p q => ⟨p.1 / q.1, p.2 / q.2⟩⟩


@[to_additive (attr := simp)]
theorem fst_div [Div G] [Div H] (a b : G × H) : (a / b).1 = a.1 / b.1 :=
  rfl


@[to_additive (attr := simp)]
theorem snd_div [Div G] [Div H] (a b : G × H) : (a / b).2 = a.2 / b.2 :=
  rfl


@[to_additive (attr := simp)]
theorem mk_div_mk [Div G] [Div H] (x₁ x₂ : G) (y₁ y₂ : H) :
    (x₁, y₁) / (x₂, y₂) = (x₁ / x₂, y₁ / y₂) :=
  rfl


@[to_additive (attr := simp)]
theorem swap_div [Div G] [Div H] (a b : G × H) : (a / b).swap = a.swap / b.swap :=
  rfl


@[to_additive] lemma div_def [Div M] [Div N] (a b : M × N) : a / b = (a.1 / b.1, a.2 / b.2) := rfl


@[to_additive]
instance instSemigroup [Semigroup M] [Semigroup N] : Semigroup (M × N) :=
  { mul_assoc := fun _ _ _ => mk.inj_iff.mpr ⟨mul_assoc _ _ _, mul_assoc _ _ _⟩ }


@[to_additive]
instance instCommSemigroup [CommSemigroup G] [CommSemigroup H] : CommSemigroup (G × H) :=
  { mul_comm := fun _ _ => mk.inj_iff.mpr ⟨mul_comm _ _, mul_comm _ _⟩ }


@[to_additive]
instance instMulOneClass [MulOneClass M] [MulOneClass N] : MulOneClass (M × N) :=
  { one_mul := fun a => Prod.recOn a fun _ _ => mk.inj_iff.mpr ⟨one_mul _, one_mul _⟩,
    mul_one := fun a => Prod.recOn a fun _ _ => mk.inj_iff.mpr ⟨mul_one _, mul_one _⟩ }


@[to_additive]
instance instMonoid [Monoid M] [Monoid N] : Monoid (M × N) :=
  { npow := fun z a => ⟨Monoid.npow z a.1, Monoid.npow z a.2⟩,
    npow_zero := fun _ => Prod.ext (Monoid.npow_zero _) (Monoid.npow_zero _),
    npow_succ := fun _ _ => Prod.ext (Monoid.npow_succ _ _) (Monoid.npow_succ _ _),
                  /-
                    G : Type u_1
                    H : Type u_2
                    M : Type u_3
                    N : Type u_4
                    P : Type u_5
                    inst✝¹ : Monoid M
                    inst✝ : Monoid N
                    ⊢ ∀ (a : Prod M N), Eq (HMul.hMul 1 a) a
                  -/
    one_mul := by simp,
                  /-
                    🎉 no goals
                  -/
                  /-
                    G : Type u_1
                    H : Type u_2
                    M : Type u_3
                    N : Type u_4
                    P : Type u_5
                    inst✝¹ : Monoid M
                    inst✝ : Monoid N
                    ⊢ ∀ (a : Prod M N), Eq (HMul.hMul a 1) a
                  -/
    mul_one := by simp }
                  /-
                    🎉 no goals
                  -/


@[to_additive Prod.subNegMonoid]
instance [DivInvMonoid G] [DivInvMonoid H] : DivInvMonoid (G × H) :=
  { div_eq_mul_inv := fun _ _ => mk.inj_iff.mpr ⟨div_eq_mul_inv _ _, div_eq_mul_inv _ _⟩,
    zpow := fun z a => ⟨DivInvMonoid.zpow z a.1, DivInvMonoid.zpow z a.2⟩,
    zpow_zero' := fun _ => Prod.ext (DivInvMonoid.zpow_zero' _) (DivInvMonoid.zpow_zero' _),
    zpow_succ' := fun _ _ => Prod.ext (DivInvMonoid.zpow_succ' _ _) (DivInvMonoid.zpow_succ' _ _),
    zpow_neg' := fun _ _ => Prod.ext (DivInvMonoid.zpow_neg' _ _) (DivInvMonoid.zpow_neg' _ _) }


@[to_additive]
instance [DivisionMonoid G] [DivisionMonoid H] : DivisionMonoid (G × H) :=
  { mul_inv_rev := fun _ _ => Prod.ext (mul_inv_rev _ _) (mul_inv_rev _ _),
    inv_eq_of_mul := fun _ _ h =>
      Prod.ext (inv_eq_of_mul_eq_one_right <| congr_arg fst h)
        (inv_eq_of_mul_eq_one_right <| congr_arg snd h),
                  /-
                    G : Type u_1
                    H : Type u_2
                    M : Type u_3
                    N : Type u_4
                    P : Type u_5
                    inst✝¹ : DivisionMonoid G
                    inst✝ : DivisionMonoid H
                    ⊢ ∀ (x : Prod G H), Eq (Inv.inv (Inv.inv x)) x
                  -/
    inv_inv := by simp }
                  /-
                    🎉 no goals
                  -/


@[to_additive SubtractionCommMonoid]
instance [DivisionCommMonoid G] [DivisionCommMonoid H] : DivisionCommMonoid (G × H) :=
                                           /-
                                             G : Type u_1
                                             H : Type u_2
                                             M : Type u_3
                                             N : Type u_4
                                             P : Type u_5
                                             inst✝¹ : DivisionCommMonoid G
                                             inst✝ : DivisionCommMonoid H
                                             x✝¹ x✝ : Prod G H
                                             g₁ : G
                                             h₁ : H
                                             fst✝ : G
                                             snd✝ : H
                                             ⊢ Eq (HMul.hMul { fst := g₁, snd := h₁ } { fst := fst✝, snd := snd✝ }) (HMul.h …
                                           -/
  { mul_comm := fun ⟨g₁ , h₁⟩ ⟨_, _⟩ => by rw [mk_mul_mk, mul_comm g₁, mul_comm h₁]; rfl }
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[to_additive]
instance instGroup [Group G] [Group H] : Group (G × H) :=
  { inv_mul_cancel := fun _ => mk.inj_iff.mpr ⟨inv_mul_cancel _, inv_mul_cancel _⟩ }


@[to_additive]
instance [Mul G] [Mul H] [IsLeftCancelMul G] [IsLeftCancelMul H] : IsLeftCancelMul (G × H) where
  mul_left_cancel _ _ _ h :=
      Prod.ext (mul_left_cancel (Prod.ext_iff.1 h).1) (mul_left_cancel (Prod.ext_iff.1 h).2)


@[to_additive]
instance [Mul G] [Mul H] [IsRightCancelMul G] [IsRightCancelMul H] : IsRightCancelMul (G × H) where
  mul_right_cancel _ _ _ h :=
      Prod.ext (mul_right_cancel (Prod.ext_iff.1 h).1) (mul_right_cancel (Prod.ext_iff.1 h).2)


@[to_additive]
instance [Mul G] [Mul H] [IsCancelMul G] [IsCancelMul H] : IsCancelMul (G × H) where


@[to_additive]
instance [LeftCancelSemigroup G] [LeftCancelSemigroup H] : LeftCancelSemigroup (G × H) :=
  { mul_left_cancel := fun _ _ _ => mul_left_cancel }


@[to_additive]
instance [RightCancelSemigroup G] [RightCancelSemigroup H] : RightCancelSemigroup (G × H) :=
  { mul_right_cancel := fun _ _ _ => mul_right_cancel }


@[to_additive]
instance [LeftCancelMonoid M] [LeftCancelMonoid N] : LeftCancelMonoid (M × N) :=
                  /-
                    G : Type u_1
                    H : Type u_2
                    M : Type u_3
                    N : Type u_4
                    P : Type u_5
                    inst✝¹ : LeftCancelMonoid M
                    inst✝ : LeftCancelMonoid N
                    ⊢ ∀ (a : Prod M N), Eq (HMul.hMul a 1) a
                  -/
                  /-
                    G : Type u_1
                    H : Type u_2
                    M : Type u_3
                    N : Type u_4
                    P : Type u_5
                    inst✝¹ : LeftCancelMonoid M
                    inst✝ : LeftCancelMonoid N
                    ⊢ ∀ (a : Prod M N), Eq (HMul.hMul 1 a) a
                  -/
  { mul_one := by simp,
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
    one_mul := by simp
                          /-
                            G : Type u_1
                            H : Type u_2
                            M : Type u_3
                            N : Type u_4
                            P : Type u_5
                            inst✝¹ : LeftCancelMonoid M
                            inst✝ : LeftCancelMonoid N
                            ⊢ ∀ (a b c : Prod M N), Eq (HMul.hMul a b) (HMul.hMul a c) → Eq b c
                          -/
    mul_left_cancel := by simp }
                          /-
                            🎉 no goals
                          -/


@[to_additive]
instance [RightCancelMonoid M] [RightCancelMonoid N] : RightCancelMonoid (M × N) :=
                  /-
                    G : Type u_1
                    H : Type u_2
                    M : Type u_3
                    N : Type u_4
                    P : Type u_5
                    inst✝¹ : RightCancelMonoid M
                    inst✝ : RightCancelMonoid N
                    ⊢ ∀ (a : Prod M N), Eq (HMul.hMul a 1) a
                  -/
                  /-
                    G : Type u_1
                    H : Type u_2
                    M : Type u_3
                    N : Type u_4
                    P : Type u_5
                    inst✝¹ : RightCancelMonoid M
                    inst✝ : RightCancelMonoid N
                    ⊢ ∀ (a : Prod M N), Eq (HMul.hMul 1 a) a
                  -/
  { mul_one := by simp,
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
    one_mul := by simp
                           /-
                             G : Type u_1
                             H : Type u_2
                             M : Type u_3
                             N : Type u_4
                             P : Type u_5
                             inst✝¹ : RightCancelMonoid M
                             inst✝ : RightCancelMonoid N
                             ⊢ ∀ (a b c : Prod M N), Eq (HMul.hMul a b) (HMul.hMul c b) → Eq a c
                           -/
    mul_right_cancel := by simp }
                           /-
                             🎉 no goals
                           -/


@[to_additive]
instance [CancelMonoid M] [CancelMonoid N] : CancelMonoid (M × N) :=
                           /-
                             G : Type u_1
                             H : Type u_2
                             M : Type u_3
                             N : Type u_4
                             P : Type u_5
                             inst✝¹ : CancelMonoid M
                             inst✝ : CancelMonoid N
                             ⊢ ∀ (a b c : Prod M N), Eq (HMul.hMul a b) (HMul.hMul c b) → Eq a c
                           -/
  { mul_right_cancel := by simp only [mul_left_inj, imp_self, forall_const] }
                           /-
                             🎉 no goals
                           -/


@[to_additive]
instance instCommMonoid [CommMonoid M] [CommMonoid N] : CommMonoid (M × N) :=
                                          /-
                                            G : Type u_1
                                            H : Type u_2
                                            M : Type u_3
                                            N : Type u_4
                                            P : Type u_5
                                            inst✝¹ : CommMonoid M
                                            inst✝ : CommMonoid N
                                            x✝¹ x✝ : Prod M N
                                            m₁ : M
                                            n₁ : N
                                            fst✝ : M
                                            snd✝ : N
                                            ⊢ Eq (HMul.hMul { fst := m₁, snd := n₁ } { fst := fst✝, snd := snd✝ }) (HMul.h …
                                          -/
  { mul_comm := fun ⟨m₁, n₁⟩ ⟨_, _⟩ => by rw [mk_mul_mk, mk_mul_mk, mul_comm m₁, mul_comm n₁] }
                                          /-
                                            🎉 no goals
                                          -/


@[to_additive]
instance [CancelCommMonoid M] [CancelCommMonoid N] : CancelCommMonoid (M × N) :=
                          /-
                            G : Type u_1
                            H : Type u_2
                            M : Type u_3
                            N : Type u_4
                            P : Type u_5
                            inst✝¹ : CancelCommMonoid M
                            inst✝ : CancelCommMonoid N
                            ⊢ ∀ (a b c : Prod M N), Eq (HMul.hMul a b) (HMul.hMul a c) → Eq b c
                          -/
  { mul_left_cancel := by simp }
                          /-
                            🎉 no goals
                          -/


@[to_additive]
instance instCommGroup [CommGroup G] [CommGroup H] : CommGroup (G × H) :=
                                          /-
                                            G : Type u_1
                                            H : Type u_2
                                            M : Type u_3
                                            N : Type u_4
                                            P : Type u_5
                                            inst✝¹ : CommGroup G
                                            inst✝ : CommGroup H
                                            x✝¹ x✝ : Prod G H
                                            g₁ : G
                                            h₁ : H
                                            fst✝ : G
                                            snd✝ : H
                                            ⊢ Eq (HMul.hMul { fst := g₁, snd := h₁ } { fst := fst✝, snd := snd✝ }) (HMul.h …
                                          -/
  { mul_comm := fun ⟨g₁, h₁⟩ ⟨_, _⟩ => by rw [mk_mul_mk, mk_mul_mk, mul_comm g₁, mul_comm h₁] }
                                          /-
                                            🎉 no goals
                                          -/


@[to_additive AddSemiconjBy.prod]
theorem SemiconjBy.prod {x y z : M × N}
    (hm : SemiconjBy x.1 y.1 z.1) (hn : SemiconjBy x.2 y.2 z.2) : SemiconjBy x y z :=
  Prod.ext hm hn


@[to_additive]
theorem Prod.semiconjBy_iff {x y z : M × N} :
    SemiconjBy x y z ↔ SemiconjBy x.1 y.1 z.1 ∧ SemiconjBy x.2 y.2 z.2 := Prod.ext_iff


@[to_additive AddCommute.prod]
theorem Commute.prod {x y : M × N} (hm : Commute x.1 y.1) (hn : Commute x.2 y.2) : Commute x y :=
  .prod hm hn


@[to_additive]
theorem Prod.commute_iff {x y : M × N} :
    Commute x y ↔ Commute x.1 y.1 ∧ Commute x.2 y.2 := semiconjBy_iff


/-- Given magmas `M`, `N`, the natural projection homomorphism from `M × N` to `M`. -/
@[to_additive
      "Given additive magmas `A`, `B`, the natural projection homomorphism
      from `A × B` to `A`"]
def fst : M × N →ₙ* M :=
  ⟨Prod.fst, fun _ _ => rfl⟩


/-- Given magmas `M`, `N`, the natural projection homomorphism from `M × N` to `N`. -/
@[to_additive
      "Given additive magmas `A`, `B`, the natural projection homomorphism
      from `A × B` to `B`"]
def snd : M × N →ₙ* N :=
  ⟨Prod.snd, fun _ _ => rfl⟩


@[to_additive (attr := simp)]
theorem coe_fst : ⇑(fst M N) = Prod.fst :=
  rfl


@[to_additive (attr := simp)]
theorem coe_snd : ⇑(snd M N) = Prod.snd :=
  rfl


/-- Combine two `MonoidHom`s `f : M →ₙ* N`, `g : M →ₙ* P` into
`f.prod g : M →ₙ* (N × P)` given by `(f.prod g) x = (f x, g x)`. -/
@[to_additive prod
      "Combine two `AddMonoidHom`s `f : AddHom M N`, `g : AddHom M P` into
      `f.prod g : AddHom M (N × P)` given by `(f.prod g) x = (f x, g x)`"]
protected def prod (f : M →ₙ* N) (g : M →ₙ* P) :
    M →ₙ* N × P where
  toFun := Pi.prod f g
  map_mul' x y := Prod.ext (f.map_mul x y) (g.map_mul x y)


@[to_additive coe_prod]
theorem coe_prod (f : M →ₙ* N) (g : M →ₙ* P) : ⇑(f.prod g) = Pi.prod f g :=
  rfl


@[to_additive (attr := simp) prod_apply]
theorem prod_apply (f : M →ₙ* N) (g : M →ₙ* P) (x) : f.prod g x = (f x, g x) :=
  rfl


@[to_additive (attr := simp) fst_comp_prod]
theorem fst_comp_prod (f : M →ₙ* N) (g : M →ₙ* P) : (fst N P).comp (f.prod g) = f :=
  ext fun _ => rfl


@[to_additive (attr := simp) snd_comp_prod]
theorem snd_comp_prod (f : M →ₙ* N) (g : M →ₙ* P) : (snd N P).comp (f.prod g) = g :=
  ext fun _ => rfl


@[to_additive (attr := simp) prod_unique]
theorem prod_unique (f : M →ₙ* N × P) : ((fst N P).comp f).prod ((snd N P).comp f) = f :=
                  /-
                    M : Type u_3
                    N : Type u_4
                    P : Type u_5
                    inst✝² : Mul M
                    inst✝¹ : Mul N
                    inst✝ : Mul P
                    f : MulHom M (Prod N P)
                    x : M
                    ⊢ Eq ((((MulHom.fst N P).comp f).prod ((MulHom.snd N P).comp f)) x) (f x)
                  -/
  ext fun x => by simp only [prod_apply, coe_fst, coe_snd, comp_apply]
                  /-
                    🎉 no goals
                  -/


/-- `Prod.map` as a `MonoidHom`. -/
@[to_additive prodMap "`Prod.map` as an `AddMonoidHom`"]
def prodMap : M × N →ₙ* M' × N' :=
  (f.comp (fst M N)).prod (g.comp (snd M N))


@[to_additive prodMap_def]
theorem prodMap_def : prodMap f g = (f.comp (fst M N)).prod (g.comp (snd M N)) :=
  rfl


@[to_additive (attr := simp) coe_prodMap]
theorem coe_prodMap : ⇑(prodMap f g) = Prod.map f g :=
  rfl


@[to_additive prod_comp_prodMap]
theorem prod_comp_prodMap (f : P →ₙ* M) (g : P →ₙ* N) (f' : M →ₙ* M') (g' : N →ₙ* N') :
    (f'.prodMap g').comp (f.prod g) = (f'.comp f).prod (g'.comp g) :=
  rfl


/-- Coproduct of two `MulHom`s with the same codomain:
  `f.coprod g (p : M × N) = f p.1 * g p.2`.
  (Commutative codomain; for the general case, see `MulHom.noncommCoprod`) -/
@[to_additive
    "Coproduct of two `AddHom`s with the same codomain:
    `f.coprod g (p : M × N) = f p.1 + g p.2`.
    (Commutative codomain; for the general case, see `AddHom.noncommCoprod`)"]
def coprod : M × N →ₙ* P :=
  f.comp (fst M N) * g.comp (snd M N)


@[to_additive (attr := simp)]
theorem coprod_apply (p : M × N) : f.coprod g p = f p.1 * g p.2 :=
  rfl


@[to_additive]
theorem comp_coprod {Q : Type*} [CommSemigroup Q] (h : P →ₙ* Q) (f : M →ₙ* P) (g : N →ₙ* P) :
    h.comp (f.coprod g) = (h.comp f).coprod (h.comp g) :=
                  /-
                    M : Type u_3
                    N : Type u_4
                    P : Type u_5
                    inst✝³ : Mul M
                    inst✝² : Mul N
                    inst✝¹ : CommSemigroup P
                    Q : Type u_6
                    inst✝ : CommSemigroup Q
                    h : MulHom P Q
                    f : MulHom M P
                    g : MulHom N P
                    x : Prod M N
                    ⊢ Eq ((h.comp (f.coprod g)) x) (((h.comp f).coprod (h.comp g)) x)
                  -/
  ext fun x => by simp
                  /-
                    🎉 no goals
                  -/


/-- Given monoids `M`, `N`, the natural projection homomorphism from `M × N` to `M`. -/
@[to_additive
      "Given additive monoids `A`, `B`, the natural projection homomorphism
      from `A × B` to `A`"]
def fst : M × N →* M :=
  { toFun := Prod.fst,
    map_one' := rfl,
    map_mul' := fun _ _ => rfl }


/-- Given monoids `M`, `N`, the natural projection homomorphism from `M × N` to `N`. -/
@[to_additive
      "Given additive monoids `A`, `B`, the natural projection homomorphism
      from `A × B` to `B`"]
def snd : M × N →* N :=
  { toFun := Prod.snd,
    map_one' := rfl,
    map_mul' := fun _ _ => rfl }


/-- Given monoids `M`, `N`, the natural inclusion homomorphism from `M` to `M × N`. -/
@[to_additive
      "Given additive monoids `A`, `B`, the natural inclusion homomorphism
      from `A` to `A × B`."]
def inl : M →* M × N :=
  { toFun := fun x => (x, 1),
    map_one' := rfl,
    map_mul' := fun _ _ => Prod.ext rfl (one_mul 1).symm }


/-- Given monoids `M`, `N`, the natural inclusion homomorphism from `N` to `M × N`. -/
@[to_additive
      "Given additive monoids `A`, `B`, the natural inclusion homomorphism
      from `B` to `A × B`."]
def inr : N →* M × N :=
  { toFun := fun y => (1, y),
    map_one' := rfl,
    map_mul' := fun _ _ => Prod.ext (one_mul 1).symm rfl }


@[to_additive (attr := simp)]
theorem inl_apply (x) : inl M N x = (x, 1) :=
  rfl


@[to_additive (attr := simp)]
theorem inr_apply (y) : inr M N y = (1, y) :=
  rfl


@[to_additive (attr := simp)]
theorem fst_comp_inl : (fst M N).comp (inl M N) = id M :=
  rfl


@[to_additive (attr := simp)]
theorem snd_comp_inl : (snd M N).comp (inl M N) = 1 :=
  rfl


@[to_additive (attr := simp)]
theorem fst_comp_inr : (fst M N).comp (inr M N) = 1 :=
  rfl


@[to_additive (attr := simp)]
theorem snd_comp_inr : (snd M N).comp (inr M N) = id N :=
  rfl


@[to_additive]
theorem commute_inl_inr (m : M) (n : N) : Commute (inl M N m) (inr M N n) :=
  Commute.prod (.one_right m) (.one_left n)


/-- Combine two `MonoidHom`s `f : M →* N`, `g : M →* P` into `f.prod g : M →* N × P`
given by `(f.prod g) x = (f x, g x)`. -/
@[to_additive prod
      "Combine two `AddMonoidHom`s `f : M →+ N`, `g : M →+ P` into
      `f.prod g : M →+ N × P` given by `(f.prod g) x = (f x, g x)`"]
protected def prod (f : M →* N) (g : M →* P) :
    M →* N × P where
  toFun := Pi.prod f g
  map_one' := Prod.ext f.map_one g.map_one
  map_mul' x y := Prod.ext (f.map_mul x y) (g.map_mul x y)


@[to_additive coe_prod]
theorem coe_prod (f : M →* N) (g : M →* P) : ⇑(f.prod g) = Pi.prod f g :=
  rfl


@[to_additive (attr := simp) prod_apply]
theorem prod_apply (f : M →* N) (g : M →* P) (x) : f.prod g x = (f x, g x) :=
  rfl


@[to_additive (attr := simp) fst_comp_prod]
theorem fst_comp_prod (f : M →* N) (g : M →* P) : (fst N P).comp (f.prod g) = f :=
  ext fun _ => rfl


@[to_additive (attr := simp) snd_comp_prod]
theorem snd_comp_prod (f : M →* N) (g : M →* P) : (snd N P).comp (f.prod g) = g :=
  ext fun _ => rfl


@[to_additive (attr := simp) prod_unique]
theorem prod_unique (f : M →* N × P) : ((fst N P).comp f).prod ((snd N P).comp f) = f :=
                  /-
                    M : Type u_3
                    N : Type u_4
                    P : Type u_5
                    inst✝² : MulOneClass M
                    inst✝¹ : MulOneClass N
                    inst✝ : MulOneClass P
                    f : MonoidHom M (Prod N P)
                    x : M
                    ⊢ Eq ((((MonoidHom.fst N P).comp f).prod ((MonoidHom.snd N P).comp f)) x) (f x)
                  -/
  ext fun x => by simp only [prod_apply, coe_fst, coe_snd, comp_apply]
                  /-
                    🎉 no goals
                  -/


/-- `prod.map` as a `MonoidHom`. -/
@[to_additive prodMap "`prod.map` as an `AddMonoidHom`."]
def prodMap : M × N →* M' × N' :=
  (f.comp (fst M N)).prod (g.comp (snd M N))


@[to_additive prod_comp_prodMap]
theorem prod_comp_prodMap (f : P →* M) (g : P →* N) (f' : M →* M') (g' : N →* N') :
    (f'.prodMap g').comp (f.prod g) = (f'.comp f).prod (g'.comp g) :=
  rfl


/-- Coproduct of two `MonoidHom`s with the same codomain:
  `f.coprod g (p : M × N) = f p.1 * g p.2`.
  (Commutative case; for the general case, see `MonoidHom.noncommCoprod`.)-/
@[to_additive
    "Coproduct of two `AddMonoidHom`s with the same codomain:
    `f.coprod g (p : M × N) = f p.1 + g p.2`.
    (Commutative case; for the general case, see `AddHom.noncommCoprod`.)"]
def coprod : M × N →* P :=
  f.comp (fst M N) * g.comp (snd M N)


@[to_additive (attr := simp)]
theorem coprod_comp_inl : (f.coprod g).comp (inl M N) = f :=
                  /-
                    M : Type u_3
                    N : Type u_4
                    P : Type u_5
                    inst✝² : MulOneClass M
                    inst✝¹ : MulOneClass N
                    inst✝ : CommMonoid P
                    f : MonoidHom M P
                    g : MonoidHom N P
                    x : M
                    ⊢ Eq (((f.coprod g).comp (MonoidHom.inl M N)) x) (f x)
                  -/
  ext fun x => by simp [coprod_apply]
                  /-
                    🎉 no goals
                  -/


@[to_additive (attr := simp)]
theorem coprod_comp_inr : (f.coprod g).comp (inr M N) = g :=
                  /-
                    M : Type u_3
                    N : Type u_4
                    P : Type u_5
                    inst✝² : MulOneClass M
                    inst✝¹ : MulOneClass N
                    inst✝ : CommMonoid P
                    f : MonoidHom M P
                    g : MonoidHom N P
                    x : N
                    ⊢ Eq (((f.coprod g).comp (MonoidHom.inr M N)) x) (g x)
                  -/
  ext fun x => by simp [coprod_apply]
                  /-
                    🎉 no goals
                  -/


@[to_additive (attr := simp)]
theorem coprod_unique (f : M × N →* P) : (f.comp (inl M N)).coprod (f.comp (inr M N)) = f :=
                  /-
                    M : Type u_3
                    N : Type u_4
                    P : Type u_5
                    inst✝² : MulOneClass M
                    inst✝¹ : MulOneClass N
                    inst✝ : CommMonoid P
                    f : MonoidHom (Prod M N) P
                    x : Prod M N
                    ⊢ Eq (((f.comp (MonoidHom.inl M N)).coprod (f.comp (MonoidHom.inr M N))) x) (f …
                  -/
  ext fun x => by simp [coprod_apply, inl_apply, inr_apply, ← map_mul]
                  /-
                    🎉 no goals
                  -/


@[to_additive (attr := simp)]
theorem coprod_inl_inr {M N : Type*} [CommMonoid M] [CommMonoid N] :
    (inl M N).coprod (inr M N) = id (M × N) :=
  coprod_unique (id <| M × N)


@[to_additive]
theorem comp_coprod {Q : Type*} [CommMonoid Q] (h : P →* Q) (f : M →* P) (g : N →* P) :
    h.comp (f.coprod g) = (h.comp f).coprod (h.comp g) :=
                  /-
                    M : Type u_3
                    N : Type u_4
                    P : Type u_5
                    inst✝³ : MulOneClass M
                    inst✝² : MulOneClass N
                    inst✝¹ : CommMonoid P
                    Q : Type u_6
                    inst✝ : CommMonoid Q
                    h : MonoidHom P Q
                    f : MonoidHom M P
                    g : MonoidHom N P
                    x : Prod M N
                    ⊢ Eq ((h.comp (f.coprod g)) x) (((h.comp f).coprod (h.comp g)) x)
                  -/
  ext fun x => by simp
                  /-
                    🎉 no goals
                  -/


/-- The equivalence between `M × N` and `N × M` given by swapping the components
is multiplicative. -/
@[to_additive prodComm
      "The equivalence between `M × N` and `N × M` given by swapping the
      components is additive."]
def prodComm : M × N ≃* N × M :=
  { Equiv.prodComm M N with map_mul' := fun ⟨_, _⟩ ⟨_, _⟩ => rfl }


@[to_additive (attr := simp) coe_prodComm]
theorem coe_prodComm : ⇑(prodComm : M × N ≃* N × M) = Prod.swap :=
  rfl


@[to_additive (attr := simp) coe_prodComm_symm]
theorem coe_prodComm_symm : ⇑(prodComm : M × N ≃* N × M).symm = Prod.swap :=
  rfl


/-- The equivalence between `(M × N) × P` and `M × (N × P)` is multiplicative. -/
@[to_additive prodAssoc
      "The equivalence between `(M × N) × P` and `M × (N × P)` is additive."]
def prodAssoc : (M × N) × P ≃* M × (N × P) :=
  { Equiv.prodAssoc M N P with map_mul' := fun ⟨_, _⟩ ⟨_, _⟩ => rfl }


@[to_additive (attr := simp) coe_prodAssoc]
theorem coe_prodAssoc : ⇑(prodAssoc : (M × N) × P ≃* M × (N × P)) = Equiv.prodAssoc M N P :=
  rfl


@[to_additive (attr := simp) coe_prodAssoc_symm]
theorem coe_prodAssoc_symm :
    ⇑(prodAssoc : (M × N) × P ≃* M × (N × P)).symm = (Equiv.prodAssoc M N P).symm :=
  rfl


/-- Four-way commutativity of `Prod`. The name matches `mul_mul_mul_comm`. -/
@[to_additive (attr := simps apply) prodProdProdComm
    "Four-way commutativity of `Prod`.\nThe name matches `mul_mul_mul_comm`"]
def prodProdProdComm : (M × N) × M' × N' ≃* (M × M') × N × N' :=
  { Equiv.prodProdProdComm M N M' N' with
    toFun := fun mnmn => ((mnmn.1.1, mnmn.2.1), (mnmn.1.2, mnmn.2.2))
    invFun := fun mmnn => ((mmnn.1.1, mmnn.2.1), (mmnn.1.2, mmnn.2.2))
    map_mul' := fun _mnmn _mnmn' => rfl }


@[to_additive (attr := simp) prodProdProdComm_toEquiv]
theorem prodProdProdComm_toEquiv :
    (prodProdProdComm M N M' N' : _ ≃ _) = Equiv.prodProdProdComm M N M' N' :=
  rfl


@[simp]
theorem prodProdProdComm_symm : (prodProdProdComm M N M' N').symm = prodProdProdComm M M' N N' :=
  rfl


/-- Product of multiplicative isomorphisms; the maps come from `Equiv.prodCongr`. -/
@[to_additive prodCongr "Product of additive isomorphisms; the maps come from `Equiv.prodCongr`."]
def prodCongr (f : M ≃* M') (g : N ≃* N') : M × N ≃* M' × N' :=
  { f.toEquiv.prodCongr g.toEquiv with
    map_mul' := fun _ _ => Prod.ext (map_mul f _ _) (map_mul g _ _) }


/-- Multiplying by the trivial monoid doesn't change the structure. -/
@[to_additive uniqueProd "Multiplying by the trivial monoid doesn't change the structure."]
def uniqueProd [Unique N] : N × M ≃* M :=
  { Equiv.uniqueProd M N with map_mul' := fun _ _ => rfl }


/-- Multiplying by the trivial monoid doesn't change the structure. -/
@[to_additive prodUnique "Multiplying by the trivial monoid doesn't change the structure."]
def prodUnique [Unique N] : M × N ≃* M :=
  { Equiv.prodUnique M N with map_mul' := fun _ _ => rfl }


/-- The monoid equivalence between units of a product of two monoids, and the product of the
    units of each monoid. -/
@[to_additive prodAddUnits
      "The additive monoid equivalence between additive units of a product
      of two additive monoids, and the product of the additive units of each additive monoid."]
def prodUnits : (M × N)ˣ ≃* Mˣ × Nˣ where
  toFun := (Units.map (MonoidHom.fst M N)).prod (Units.map (MonoidHom.snd M N))
                                                /-
                                                  G : Type u_1
                                                  H : Type u_2
                                                  M : Type u_3
                                                  N : Type u_4
                                                  P : Type u_5
                                                  inst✝¹ : Monoid M
                                                  inst✝ : Monoid N
                                                  u : Prod (Units M) (Units N)
                                                  ⊢ Eq (HMul.hMul { fst := ↑u.1, snd := ↑u.2 } { fst := ↑(Inv.inv u.1), snd := ↑ …
                                                -/
                                                /-
                                                  🎉 no goals
                                                -/
  invFun u := ⟨(u.1, u.2), (↑u.1⁻¹, ↑u.2⁻¹), by simp, by simp⟩
                                                         /-
                                                           🎉 no goals
                                                         -/
  left_inv u := by
    simp only [MonoidHom.prod_apply, Units.coe_map, MonoidHom.coe_fst, MonoidHom.coe_snd,
      Prod.mk.eta, Units.coe_map_inv, Units.mk_val]
  right_inv := fun ⟨u₁, u₂⟩ => by
    simp only [Units.map, MonoidHom.coe_fst, Units.inv_eq_val_inv,
      MonoidHom.coe_snd, MonoidHom.prod_apply, Prod.mk.injEq]
    /-
      G : Type u_1
      H : Type u_2
      M : Type u_3
      N : Type u_4
      P : Type u_5
      inst✝¹ : Monoid M
      inst✝ : Monoid N
      x✝ : Prod (Units M) (Units N)
      u₁ : Units M
      u₂ : Units N
      ⊢ And (Eq ((MonoidHom.mk' (fun u => { val := (↑u).1, inv := (↑(Inv.inv u)).1,  …
    -/
    exact ⟨rfl, rfl⟩
    /-
      🎉 no goals
    -/
  map_mul' := MonoidHom.map_mul _


@[to_additive]
lemma _root_.Prod.isUnit_iff {x : M × N} : IsUnit x ↔ IsUnit x.1 ∧ IsUnit x.2 where
  mp h := ⟨(prodUnits h.unit).1.isUnit, (prodUnits h.unit).2.isUnit⟩
  mpr h := (prodUnits.symm (h.1.unit, h.2.unit)).isUnit


/-- Canonical homomorphism of monoids from `αˣ` into `α × αᵐᵒᵖ`.
Used mainly to define the natural topology of `αˣ`. -/
@[to_additive (attr := simps)
      "Canonical homomorphism of additive monoids from `AddUnits α` into `α × αᵃᵒᵖ`.
      Used mainly to define the natural topology of `AddUnits α`."]
def embedProduct (α : Type*) [Monoid α] : αˣ →* α × αᵐᵒᵖ where
  toFun x := ⟨x, op ↑x⁻¹⟩
  map_one' := by
    /-
      G : Type u_1
      H : Type u_2
      M : Type u_3
      N : Type u_4
      P : Type u_5
      α : Type u_6
      inst✝ : Monoid α
      ⊢ Eq ((fun x => { fst := ↑x, snd := MulOpposite.op ↑(Inv.inv x) }) 1) 1
    -/
    simp only [inv_one, eq_self_iff_true, Units.val_one, op_one, Prod.mk_eq_one, and_self_iff]
    /-
      🎉 no goals
    -/
                     /-
                       G : Type u_1
                       H : Type u_2
                       M : Type u_3
                       N : Type u_4
                       P : Type u_5
                       α : Type u_6
                       inst✝ : Monoid α
                       x y : Units α
                       ⊢ Eq ({ toFun := fun x => { fst := ↑x, snd := MulOpposite.op ↑(Inv.inv x) }, m …
                     -/
  map_mul' x y := by simp only [mul_inv_rev, op_mul, Units.val_mul, Prod.mk_mul_mk]
                     /-
                       🎉 no goals
                     -/


@[to_additive]
theorem embedProduct_injective (α : Type*) [Monoid α] : Function.Injective (embedProduct α) :=
  fun _ _ h => Units.ext <| (congr_arg Prod.fst h : _)


/-- Multiplication as a multiplicative homomorphism. -/
@[to_additive (attr := simps) "Addition as an additive homomorphism."]
def mulMulHom [CommSemigroup α] :
    α × α →ₙ* α where
  toFun a := a.1 * a.2
  map_mul' _ _ := mul_mul_mul_comm _ _ _ _


/-- Multiplication as a monoid homomorphism. -/
@[to_additive (attr := simps) "Addition as an additive monoid homomorphism."]
def mulMonoidHom [CommMonoid α] : α × α →* α :=
  { mulMulHom with map_one' := mul_one _ }


/-- Division as a monoid homomorphism. -/
@[to_additive (attr := simps) "Subtraction as an additive monoid homomorphism."]
def divMonoidHom [DivisionCommMonoid α] : α × α →* α where
  toFun a := a.1 / a.2
  map_one' := div_one _
  map_mul' _ _ := mul_div_mul_comm _ _ _ _


