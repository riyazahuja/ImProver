/-- Coproduct of two `MulHom`s with the same codomain with `Commute` assumption:
  `f.noncommCoprod g _ (p : M × N) = f p.1 * g p.2`.
  (For the commutative case, use `MulHom.coprod`) -/
@[to_additive (attr := simps)
    "Coproduct of two `AddHom`s with the same codomain with `AddCommute` assumption:
    `f.noncommCoprod g _ (p : M × N) = f p.1 + g p.2`.
    (For the commutative case, use `AddHom.coprod`)"]
def noncommCoprod (comm : ∀ m n, Commute (f m) (g n)) : M × N →ₙ* P where
  toFun mn := f mn.fst * g mn.snd
                        /-
                          M : Type u_1
                          N : Type u_2
                          P : Type u_3
                          inst✝² : Mul M
                          inst✝¹ : Mul N
                          inst✝ : Semigroup P
                          f : MulHom M P
                          g : MulHom N P
                          comm : ∀ (m : M) (n : N), Commute (f m) (g n)
                          mn mn' : Prod M N
                          ⊢ Eq ((fun mn => HMul.hMul (f mn.1) (g mn.2)) (HMul.hMul mn mn')) (HMul.hMul ( …
                        -/
  map_mul' mn mn' := by simpa using (comm _ _).mul_mul_mul_comm _ _
                        /-
                          🎉 no goals
                        -/


/-- Variant of `MulHom.noncommCoprod_apply` with the product written in the other direction` -/
@[to_additive
  "Variant of `AddHom.noncommCoprod_apply`, with the sum written in the other direction"]
theorem noncommCoprod_apply' (comm) (mn : M × N) :
    (f.noncommCoprod g comm) mn = g mn.2 * f mn.1 := by
  /-
    M : Type u_1
    N : Type u_2
    P : Type u_3
    inst✝² : Mul M
    inst✝¹ : Mul N
    inst✝ : Semigroup P
    f : MulHom M P
    g : MulHom N P
    comm : ∀ (m : M) (n : N), Commute (f m) (g n)
    mn : Prod M N
    ⊢ Eq ((f.noncommCoprod g comm) mn) (HMul.hMul (g mn.2) (f mn.1))
  -/
  rw [← comm, noncommCoprod_apply]
  /-
    🎉 no goals
  -/



@[to_additive]
theorem comp_noncommCoprod {Q : Type*} [Semigroup Q] (h : P →ₙ* Q)
    (comm : ∀ m n, Commute (f m) (g n)) :
    h.comp (f.noncommCoprod g comm) =
      (h.comp f).noncommCoprod (h.comp g) (fun m n ↦ (comm m n).map h) :=
  ext fun _ => map_mul h _ _


/-- Coproduct of two `MonoidHom`s with the same codomain,
  with a commutation assumption:
  `f.noncommCoprod g _ (p : M × N) = f p.1 * g p.2`.
  (Noncommutative case; in the commutative case, use `MonoidHom.coprod`.)-/
@[to_additive (attr := simps)
    "Coproduct of two `AddMonoidHom`s with the same codomain,
    with a commutation assumption:
    `f.noncommCoprod g (p : M × N) = f p.1 + g p.2`.
    (Noncommutative case; in the commutative case, use `AddHom.coprod`.)"]
def noncommCoprod : M × N →* P where
  toFun := fun mn ↦ (f mn.fst) * (g mn.snd)
                 /-
                   M : Type u_1
                   N : Type u_2
                   P : Type u_3
                   inst✝² : MulOneClass M
                   inst✝¹ : MulOneClass N
                   inst✝ : Monoid P
                   f : MonoidHom M P
                   g : MonoidHom N P
                   comm : ∀ (m : M) (n : N), Commute (f m) (g n)
                   ⊢ Eq ((fun mn => HMul.hMul (f mn.1) (g mn.2)) 1) 1
                 -/
  map_one' := by simp only [Prod.fst_one, Prod.snd_one, map_one, mul_one]
                 /-
                   🎉 no goals
                 -/
  __ := f.toMulHom.noncommCoprod g.toMulHom comm


/-- Variant of `MonoidHom.noncomCoprod_apply` with the product written in the other direction` -/
@[to_additive
  "Variant of `AddMonoidHom.noncomCoprod_apply` with the sum written in the other direction"]
theorem noncommCoprod_apply' (comm) (mn : M × N) :
    (f.noncommCoprod g comm) mn = g mn.2 * f mn.1 := by
  /-
    M : Type u_1
    N : Type u_2
    P : Type u_3
    inst✝² : MulOneClass M
    inst✝¹ : MulOneClass N
    inst✝ : Monoid P
    f : MonoidHom M P
    g : MonoidHom N P
    comm : ∀ (m : M) (n : N), Commute (f m) (g n)
    mn : Prod M N
    ⊢ Eq ((f.noncommCoprod g comm) mn) (HMul.hMul (g mn.2) (f mn.1))
  -/
  rw [← comm, MonoidHom.noncommCoprod_apply]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem noncommCoprod_comp_inl : (f.noncommCoprod g comm).comp (inl M N) = f :=
                  /-
                    M : Type u_1
                    N : Type u_2
                    P : Type u_3
                    inst✝² : MulOneClass M
                    inst✝¹ : MulOneClass N
                    inst✝ : Monoid P
                    f : MonoidHom M P
                    g : MonoidHom N P
                    comm : ∀ (m : M) (n : N), Commute (f m) (g n)
                    x : M
                    ⊢ Eq (((f.noncommCoprod g comm).comp (MonoidHom.inl M N)) x) (f x)
                  -/
  ext fun x => by simp
                  /-
                    🎉 no goals
                  -/


@[to_additive (attr := simp)]
theorem noncommCoprod_comp_inr : (f.noncommCoprod g comm).comp (inr M N) = g :=
                  /-
                    M : Type u_1
                    N : Type u_2
                    P : Type u_3
                    inst✝² : MulOneClass M
                    inst✝¹ : MulOneClass N
                    inst✝ : Monoid P
                    f : MonoidHom M P
                    g : MonoidHom N P
                    comm : ∀ (m : M) (n : N), Commute (f m) (g n)
                    x : N
                    ⊢ Eq (((f.noncommCoprod g comm).comp (MonoidHom.inr M N)) x) (g x)
                  -/
  ext fun x => by simp
                  /-
                    🎉 no goals
                  -/


@[to_additive (attr := simp)]
theorem noncommCoprod_unique (f : M × N →* P) :
    (f.comp (inl M N)).noncommCoprod (f.comp (inr M N)) (fun _ _ => (commute_inl_inr _ _).map f)
      = f :=
                  /-
                    M : Type u_1
                    N : Type u_2
                    P : Type u_3
                    inst✝² : MulOneClass M
                    inst✝¹ : MulOneClass N
                    inst✝ : Monoid P
                    f : MonoidHom (Prod M N) P
                    x : Prod M N
                    ⊢ Eq (((f.comp (MonoidHom.inl M N)).noncommCoprod (f.comp (MonoidHom.inr M N)) …
                  -/
  ext fun x => by simp [coprod_apply, inl_apply, inr_apply, ← map_mul]
                  /-
                    🎉 no goals
                  -/


@[to_additive (attr := simp)]
theorem noncommCoprod_inl_inr {M N : Type*} [Monoid M] [Monoid N] :
    (inl M N).noncommCoprod (inr M N) commute_inl_inr = id (M × N) :=
  noncommCoprod_unique <| .id (M × N)


@[to_additive]
theorem comp_noncommCoprod {Q : Type*} [Monoid Q] (h : P →* Q) :
    h.comp (f.noncommCoprod g comm) =
      (h.comp f).noncommCoprod (h.comp g) (fun m n ↦ (comm m n).map h) :=
                  /-
                    M : Type u_1
                    N : Type u_2
                    P : Type u_3
                    inst✝³ : MulOneClass M
                    inst✝² : MulOneClass N
                    inst✝¹ : Monoid P
                    f : MonoidHom M P
                    g : MonoidHom N P
                    comm : ∀ (m : M) (n : N), Commute (f m) (g n)
                    Q : Type u_4
                    inst✝ : Monoid Q
                    h : MonoidHom P Q
                    x : Prod M N
                    ⊢ Eq ((h.comp (f.noncommCoprod g comm)) x) (((h.comp f).noncommCoprod (h.comp  …
                  -/
  ext fun x => by simp
                  /-
                    🎉 no goals
                  -/


