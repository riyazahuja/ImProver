/-- The equivalence relation on `M × S` where `(m1, s1) ≈ (m2, s2)` if and only if
for some (u : S), u * (s2 • m1 - s1 • m2) = 0-/
/- Porting note: We use small letter `r` since `R` is used for a ring. -/
def r (a b : M × S) : Prop :=
  ∃ u : S, u • b.2 • a.1 = u • a.2 • b.1


theorem r.isEquiv : IsEquiv _ (r S M) :=
                                 /-
                                   R : Type u
                                   inst✝² : CommSemiring R
                                   S : Submonoid R
                                   M : Type v
                                   inst✝¹ : AddCommMonoid M
                                   inst✝ : Module R M
                                   x✝ : Prod M (Subtype fun x => Membership.mem S x)
                                   m : M
                                   s : Subtype fun x => Membership.mem S x
                                   ⊢ Eq (HSMul.hSMul 1 (HSMul.hSMul { fst := m, snd := s }.2 { fst := m, snd := s …
                                 -/
  { refl := fun ⟨m, s⟩ => ⟨1, by rw [one_smul]⟩
                                 /-
                                   🎉 no goals
                                 -/
    trans := fun ⟨m1, s1⟩ ⟨m2, s2⟩ ⟨m3, s3⟩ ⟨u1, hu1⟩ ⟨u2, hu2⟩ => by
      /-
        R : Type u
        inst✝² : CommSemiring R
        S : Submonoid R
        M : Type v
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        x✝⁴ x✝³ x✝² : Prod M (Subtype fun x => Membership.mem S x)
        m1 : M
        s1 : Subtype fun x => Membership.mem S x
        m2 : M
        s2 : Subtype fun x => Membership.mem S x
        x✝¹ : LocalizedModule.r S M { fst := m1, snd := s1 } { fst := m2, snd := s2 }
        m3 : M
        s3 : Subtype fun x => Membership.mem S x
        x✝ : LocalizedModule.r S M { fst := m2, snd := s2 } { fst := m3, snd := s3 }
        u1 : Subtype fun x => Membership.mem S x
        hu1 : Eq (HSMul.hSMul u1 (HSMul.hSMul { fst := m2, snd := s2 }.2 { fst := m1,  …
        u2 : Subtype fun x => Membership.mem S x
        hu2 : Eq (HSMul.hSMul u2 (HSMul.hSMul { fst := m3, snd := s3 }.2 { fst := m2,  …
        ⊢ LocalizedModule.r S M { fst := m1, snd := s1 } { fst := m3, snd := s3 }
      -/
      use u1 * u2 * s2
      -- Put everything in the same shape, sorting the terms using `simp`
      /-
        case h
        R : Type u
        inst✝² : CommSemiring R
        S : Submonoid R
        M : Type v
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        x✝⁴ x✝³ x✝² : Prod M (Subtype fun x => Membership.mem S x)
        m1 : M
        s1 : Subtype fun x => Membership.mem S x
        m2 : M
        s2 : Subtype fun x => Membership.mem S x
        x✝¹ : LocalizedModule.r S M { fst := m1, snd := s1 } { fst := m2, snd := s2 }
        m3 : M
        s3 : Subtype fun x => Membership.mem S x
        x✝ : LocalizedModule.r S M { fst := m2, snd := s2 } { fst := m3, snd := s3 }
        u1 : Subtype fun x => Membership.mem S x
        hu1 : Eq (HSMul.hSMul u1 (HSMul.hSMul { fst := m2, snd := s2 }.2 { fst := m1,  …
        u2 : Subtype fun x => Membership.mem S x
        hu2 : Eq (HSMul.hSMul u2 (HSMul.hSMul { fst := m3, snd := s3 }.2 { fst := m2,  …
        ⊢ Eq (HSMul.hSMul (HMul.hMul (HMul.hMul u1 u2) s2) (HSMul.hSMul { fst := m3, s …
      -/
      have hu1' := congr_arg ((u2 * s3) • ·) hu1.symm
      /-
        case h
        R : Type u
        inst✝² : CommSemiring R
        S : Submonoid R
        M : Type v
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        x✝⁴ x✝³ x✝² : Prod M (Subtype fun x => Membership.mem S x)
        m1 : M
        s1 : Subtype fun x => Membership.mem S x
        m2 : M
        s2 : Subtype fun x => Membership.mem S x
        x✝¹ : LocalizedModule.r S M { fst := m1, snd := s1 } { fst := m2, snd := s2 }
        m3 : M
        s3 : Subtype fun x => Membership.mem S x
        x✝ : LocalizedModule.r S M { fst := m2, snd := s2 } { fst := m3, snd := s3 }
        u1 : Subtype fun x => Membership.mem S x
        hu1 : Eq (HSMul.hSMul u1 (HSMul.hSMul { fst := m2, snd := s2 }.2 { fst := m1,  …
        u2 : Subtype fun x => Membership.mem S x
        hu2 : Eq (HSMul.hSMul u2 (HSMul.hSMul { fst := m3, snd := s3 }.2 { fst := m2,  …
        hu1' : Eq ((fun x => HSMul.hSMul (HMul.hMul u2 s3) x) (HSMul.hSMul u1 (HSMul.h …
        ⊢ Eq (HSMul.hSMul (HMul.hMul (HMul.hMul u1 u2) s2) (HSMul.hSMul { fst := m3, s …
      -/
      have hu2' := congr_arg ((u1 * s1) • ·) hu2.symm
      /-
        case h
        R : Type u
        inst✝² : CommSemiring R
        S : Submonoid R
        M : Type v
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        x✝⁴ x✝³ x✝² : Prod M (Subtype fun x => Membership.mem S x)
        m1 : M
        s1 : Subtype fun x => Membership.mem S x
        m2 : M
        s2 : Subtype fun x => Membership.mem S x
        x✝¹ : LocalizedModule.r S M { fst := m1, snd := s1 } { fst := m2, snd := s2 }
        m3 : M
        s3 : Subtype fun x => Membership.mem S x
        x✝ : LocalizedModule.r S M { fst := m2, snd := s2 } { fst := m3, snd := s3 }
        u1 : Subtype fun x => Membership.mem S x
        hu1 : Eq (HSMul.hSMul u1 (HSMul.hSMul { fst := m2, snd := s2 }.2 { fst := m1,  …
        u2 : Subtype fun x => Membership.mem S x
        hu2 : Eq (HSMul.hSMul u2 (HSMul.hSMul { fst := m3, snd := s3 }.2 { fst := m2,  …
        hu1' : Eq ((fun x => HSMul.hSMul (HMul.hMul u2 s3) x) (HSMul.hSMul u1 (HSMul.h …
        hu2' : Eq ((fun x => HSMul.hSMul (HMul.hMul u1 s1) x) (HSMul.hSMul u2 (HSMul.h …
        ⊢ Eq (HSMul.hSMul (HMul.hMul (HMul.hMul u1 u2) s2) (HSMul.hSMul { fst := m3, s …
      -/
      simp only [← mul_smul, smul_assoc, mul_assoc, mul_comm, mul_left_comm] at hu1' hu2' ⊢
      /-
        case h
        R : Type u
        inst✝² : CommSemiring R
        S : Submonoid R
        M : Type v
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        x✝⁴ x✝³ x✝² : Prod M (Subtype fun x => Membership.mem S x)
        m1 : M
        s1 : Subtype fun x => Membership.mem S x
        m2 : M
        s2 : Subtype fun x => Membership.mem S x
        x✝¹ : LocalizedModule.r S M { fst := m1, snd := s1 } { fst := m2, snd := s2 }
        m3 : M
        s3 : Subtype fun x => Membership.mem S x
        x✝ : LocalizedModule.r S M { fst := m2, snd := s2 } { fst := m3, snd := s3 }
        u1 : Subtype fun x => Membership.mem S x
        hu1 : Eq (HSMul.hSMul u1 (HSMul.hSMul { fst := m2, snd := s2 }.2 { fst := m1,  …
        u2 : Subtype fun x => Membership.mem S x
        hu2 : Eq (HSMul.hSMul u2 (HSMul.hSMul { fst := m3, snd := s3 }.2 { fst := m2,  …
        hu1' : Eq (HSMul.hSMul (HMul.hMul s1 (HMul.hMul s3 (HMul.hMul u1 u2))) m2) (HS …
        hu2' : Eq (HSMul.hSMul (HMul.hMul s1 (HMul.hMul s2 (HMul.hMul u1 u2))) m3) (HS …
        ⊢ Eq (HSMul.hSMul (HMul.hMul s2 (HMul.hMul s3 (HMul.hMul u1 u2))) m1) (HSMul.h …
      -/
      rw [hu2', hu1']
      /-
        🎉 no goals
      -/
    symm := fun ⟨_, _⟩ ⟨_, _⟩ ⟨u, hu⟩ => ⟨u, hu.symm⟩ }


instance r.setoid : Setoid (M × S) where
  r := r S M
  iseqv := ⟨(r.isEquiv S M).refl, (r.isEquiv S M).symm _ _, (r.isEquiv S M).trans _ _ _⟩

-- TODO: change `Localization` to use `r'` instead of `r` so that the two types are also defeq,
-- `Localization S = LocalizedModule S R`.

/-- If `S` is a multiplicative subset of a ring `R` and `M` an `R`-module, then
we can localize `M` by `S`.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): @[nolint has_nonempty_instance]
def _root_.LocalizedModule : Type max u v :=
  Quotient (r.setoid S M)


/-- The canonical map sending `(m, s) ↦ m/s`-/
def mk (m : M) (s : S) : LocalizedModule S M :=
  Quotient.mk' ⟨m, s⟩


theorem mk_eq {m m' : M} {s s' : S} : mk m s = mk m' s' ↔ ∃ u : S, u • s' • m = u • s • m' :=
  Quotient.eq'


@[elab_as_elim, induction_eliminator, cases_eliminator]
theorem induction_on {β : LocalizedModule S M → Prop} (h : ∀ (m : M) (s : S), β (mk m s)) :
    ∀ x : LocalizedModule S M, β x := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    β : LocalizedModule S M → Prop
    h : ∀ (m : M) (s : Subtype fun x => Membership.mem S x), β (LocalizedModule.mk …
    ⊢ ∀ (x : LocalizedModule S M), β x
  -/
  rintro ⟨⟨m, s⟩⟩
  /-
    case mk.mk
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    β : LocalizedModule S M → Prop
    h : ∀ (m : M) (s : Subtype fun x => Membership.mem S x), β (LocalizedModule.mk …
    x✝ : LocalizedModule S M
    m : M
    s : Subtype fun x => Membership.mem S x
    ⊢ β (Quot.mk ⇑(LocalizedModule.r.setoid S M) { fst := m, snd := s })
  -/
  exact h m s
  /-
    🎉 no goals
  -/


@[elab_as_elim]
theorem induction_on₂ {β : LocalizedModule S M → LocalizedModule S M → Prop}
    (h : ∀ (m m' : M) (s s' : S), β (mk m s) (mk m' s')) : ∀ x y, β x y := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    β : LocalizedModule S M → LocalizedModule S M → Prop
    h : ∀ (m m' : M) (s s' : Subtype fun x => Membership.mem S x), β (LocalizedMod …
    ⊢ ∀ (x y : LocalizedModule S M), β x y
  -/
  rintro ⟨⟨m, s⟩⟩ ⟨⟨m', s'⟩⟩
  /-
    case mk.mk.mk.mk
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    β : LocalizedModule S M → LocalizedModule S M → Prop
    h : ∀ (m m' : M) (s s' : Subtype fun x => Membership.mem S x), β (LocalizedMod …
    x✝ : LocalizedModule S M
    m : M
    s : Subtype fun x => Membership.mem S x
    y✝ : LocalizedModule S M
    m' : M
    s' : Subtype fun x => Membership.mem S x
    ⊢ β (Quot.mk ⇑(LocalizedModule.r.setoid S M) { fst := m, snd := s }) (Quot.mk  …
  -/
  exact h m m' s s'
  /-
    🎉 no goals
  -/


/-- If `f : M × S → α` respects the equivalence relation `LocalizedModule.r`, then
`f` descents to a map `LocalizedModule M S → α`.
-/
def liftOn {α : Type*} (x : LocalizedModule S M) (f : M × S → α)
    (wd : ∀ (p p' : M × S), p ≈ p' → f p = f p') : α :=
  Quotient.liftOn x f wd


theorem liftOn_mk {α : Type*} {f : M × S → α} (wd : ∀ (p p' : M × S), p ≈ p' → f p = f p')
                                                            /-
                                                              R : Type u
                                                              inst✝² : CommSemiring R
                                                              S : Submonoid R
                                                              M : Type v
                                                              inst✝¹ : AddCommMonoid M
                                                              inst✝ : Module R M
                                                              α : Type u_2
                                                              f : Prod M (Subtype fun x => Membership.mem S x) → α
                                                              wd : ∀ (p p' : Prod M (Subtype fun x => Membership.mem S x)), HasEquiv.Equiv p …
                                                              m : M
                                                              s : Subtype fun x => Membership.mem S x
                                                              ⊢ Eq ((LocalizedModule.mk m s).liftOn f wd) (f { fst := m, snd := s })
                                                            -/
    (m : M) (s : S) : liftOn (mk m s) f wd = f ⟨m, s⟩ := by convert Quotient.liftOn_mk f wd ⟨m, s⟩
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- If `f : M × S → M × S → α` respects the equivalence relation `LocalizedModule.r`, then
`f` descents to a map `LocalizedModule M S → LocalizedModule M S → α`.
-/
def liftOn₂ {α : Type*} (x y : LocalizedModule S M) (f : M × S → M × S → α)
    (wd : ∀ (p q p' q' : M × S), p ≈ p' → q ≈ q' → f p q = f p' q') : α :=
  Quotient.liftOn₂ x y f wd


theorem liftOn₂_mk {α : Type*} (f : M × S → M × S → α)
    (wd : ∀ (p q p' q' : M × S), p ≈ p' → q ≈ q' → f p q = f p' q') (m m' : M)
    (s s' : S) : liftOn₂ (mk m s) (mk m' s') f wd = f ⟨m, s⟩ ⟨m', s'⟩ := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    α : Type u_2
    f : Prod M (Subtype fun x => Membership.mem S x) → Prod M (Subtype fun x => Me …
    wd : ∀ (p q p' q' : Prod M (Subtype fun x => Membership.mem S x)), HasEquiv.Eq …
    m m' : M
    s s' : Subtype fun x => Membership.mem S x
    ⊢ Eq ((LocalizedModule.mk m s).liftOn₂ (LocalizedModule.mk m' s') f wd) (f { f …
  -/
  convert Quotient.liftOn₂_mk f wd _ _
  /-
    🎉 no goals
  -/


instance : Zero (LocalizedModule S M) :=
  ⟨mk 0 1⟩


/-- If `S` contains `0` then the localization at `S` is trivial. -/
theorem subsingleton (h : 0 ∈ S) : Subsingleton (LocalizedModule S M) := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    h : Membership.mem S 0
    ⊢ Subsingleton (LocalizedModule S M)
  -/
  refine ⟨fun a b ↦ ?_⟩
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    h : Membership.mem S 0
    a b : LocalizedModule S M
    ⊢ Eq a b
  -/
  induction a,b using LocalizedModule.induction_on₂
  /-
    case h
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    h : Membership.mem S 0
    m✝ m'✝ : M
    s✝ s'✝ : Subtype fun x => Membership.mem S x
    ⊢ Eq (LocalizedModule.mk m✝ s✝) (LocalizedModule.mk m'✝ s'✝)
  -/
  exact mk_eq.mpr ⟨⟨0, h⟩, by simp only [Submonoid.mk_smul, zero_smul]⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_mk (s : S) : mk (0 : M) s = 0 :=
                   /-
                     R : Type u
                     inst✝² : CommSemiring R
                     S : Submonoid R
                     M : Type v
                     inst✝¹ : AddCommMonoid M
                     inst✝ : Module R M
                     s : Subtype fun x => Membership.mem S x
                     ⊢ Eq (HSMul.hSMul 1 (HSMul.hSMul 1 0)) (HSMul.hSMul 1 (HSMul.hSMul s 0))
                   -/
  mk_eq.mpr ⟨1, by rw [one_smul, smul_zero, smul_zero, one_smul]⟩
                   /-
                     🎉 no goals
                   -/


instance : Add (LocalizedModule S M) where
  add p1 p2 :=
    liftOn₂ p1 p2 (fun x y => mk (y.2 • x.1 + x.2 • y.1) (x.2 * y.2)) <|
      fun ⟨m1, s1⟩ ⟨m2, s2⟩ ⟨m1', s1'⟩ ⟨m2', s2'⟩ ⟨u1, hu1⟩ ⟨u2, hu2⟩ =>
          mk_eq.mpr
            ⟨u1 * u2, by
              -- Put everything in the same shape, sorting the terms using `simp`
              /-
                R : Type u
                inst✝⁵ : CommSemiring R
                S : Submonoid R
                M : Type v
                inst✝⁴ : AddCommMonoid M
                inst✝³ : Module R M
                T : Type u_1
                inst✝² : CommSemiring T
                inst✝¹ : Algebra R T
                inst✝ : IsLocalization S T
                p1 p2 : LocalizedModule S M
                x✝⁵ x✝⁴ x✝³ x✝² : Prod M (Subtype fun x => Membership.mem S x)
                m1 : M
                s1 : Subtype fun x => Membership.mem S x
                m2 : M
                s2 : Subtype fun x => Membership.mem S x
                m1' : M
                s1' : Subtype fun x => Membership.mem S x
                x✝¹ : HasEquiv.Equiv { fst := m1, snd := s1 } { fst := m1', snd := s1' }
                m2' : M
                s2' : Subtype fun x => Membership.mem S x
                x✝ : HasEquiv.Equiv { fst := m2, snd := s2 } { fst := m2', snd := s2' }
                u1 : Subtype fun x => Membership.mem S x
                hu1 : Eq (HSMul.hSMul u1 (HSMul.hSMul { fst := m1', snd := s1' }.2 { fst := m1 …
                u2 : Subtype fun x => Membership.mem S x
                hu2 : Eq (HSMul.hSMul u2 (HSMul.hSMul { fst := m2', snd := s2' }.2 { fst := m2 …
                ⊢ Eq (HSMul.hSMul (HMul.hMul u1 u2) (HSMul.hSMul (HMul.hMul { fst := m1', snd  …
              -/
              have hu1' := congr_arg ((u2 * s2 * s2') • ·) hu1
              /-
                R : Type u
                inst✝⁵ : CommSemiring R
                S : Submonoid R
                M : Type v
                inst✝⁴ : AddCommMonoid M
                inst✝³ : Module R M
                T : Type u_1
                inst✝² : CommSemiring T
                inst✝¹ : Algebra R T
                inst✝ : IsLocalization S T
                p1 p2 : LocalizedModule S M
                x✝⁵ x✝⁴ x✝³ x✝² : Prod M (Subtype fun x => Membership.mem S x)
                m1 : M
                s1 : Subtype fun x => Membership.mem S x
                m2 : M
                s2 : Subtype fun x => Membership.mem S x
                m1' : M
                s1' : Subtype fun x => Membership.mem S x
                x✝¹ : HasEquiv.Equiv { fst := m1, snd := s1 } { fst := m1', snd := s1' }
                m2' : M
                s2' : Subtype fun x => Membership.mem S x
                x✝ : HasEquiv.Equiv { fst := m2, snd := s2 } { fst := m2', snd := s2' }
                u1 : Subtype fun x => Membership.mem S x
                hu1 : Eq (HSMul.hSMul u1 (HSMul.hSMul { fst := m1', snd := s1' }.2 { fst := m1 …
                u2 : Subtype fun x => Membership.mem S x
                hu2 : Eq (HSMul.hSMul u2 (HSMul.hSMul { fst := m2', snd := s2' }.2 { fst := m2 …
                hu1' : Eq ((fun x => HSMul.hSMul (HMul.hMul (HMul.hMul u2 s2) s2') x) (HSMul.h …
                ⊢ Eq (HSMul.hSMul (HMul.hMul u1 u2) (HSMul.hSMul (HMul.hMul { fst := m1', snd  …
              -/
              have hu2' := congr_arg ((u1 * s1 * s1') • ·) hu2
              simp only [smul_add, ← mul_smul, smul_assoc, mul_assoc, mul_comm,
                mul_left_comm] at hu1' hu2' ⊢
              /-
                R : Type u
                inst✝⁵ : CommSemiring R
                S : Submonoid R
                M : Type v
                inst✝⁴ : AddCommMonoid M
                inst✝³ : Module R M
                T : Type u_1
                inst✝² : CommSemiring T
                inst✝¹ : Algebra R T
                inst✝ : IsLocalization S T
                p1 p2 : LocalizedModule S M
                x✝⁵ x✝⁴ x✝³ x✝² : Prod M (Subtype fun x => Membership.mem S x)
                m1 : M
                s1 : Subtype fun x => Membership.mem S x
                m2 : M
                s2 : Subtype fun x => Membership.mem S x
                m1' : M
                s1' : Subtype fun x => Membership.mem S x
                x✝¹ : HasEquiv.Equiv { fst := m1, snd := s1 } { fst := m1', snd := s1' }
                m2' : M
                s2' : Subtype fun x => Membership.mem S x
                x✝ : HasEquiv.Equiv { fst := m2, snd := s2 } { fst := m2', snd := s2' }
                u1 : Subtype fun x => Membership.mem S x
                hu1 : Eq (HSMul.hSMul u1 (HSMul.hSMul { fst := m1', snd := s1' }.2 { fst := m1 …
                u2 : Subtype fun x => Membership.mem S x
                hu2 : Eq (HSMul.hSMul u2 (HSMul.hSMul { fst := m2', snd := s2' }.2 { fst := m2 …
                hu1' : Eq (HSMul.hSMul (HMul.hMul s2 (HMul.hMul s1' (HMul.hMul s2' (HMul.hMul  …
                hu2' : Eq (HSMul.hSMul (HMul.hMul s1 (HMul.hMul s1' (HMul.hMul s2' (HMul.hMul  …
                ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HMul.hMul s2 (HMul.hMul s1' (HMul.hMul s2' (HMul …
              -/
              rw [hu1', hu2']⟩
              /-
                🎉 no goals
              -/


theorem mk_add_mk {m1 m2 : M} {s1 s2 : S} :
    mk m1 s1 + mk m2 s2 = mk (s2 • m1 + s1 • m2) (s1 * s2) :=
  mk_eq.mpr <| ⟨1, rfl⟩


private theorem add_assoc' (x y z : LocalizedModule S M) : x + y + z = x + (y + z) := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x y z : LocalizedModule S M
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd x y) z) (HAdd.hAdd x (HAdd.hAdd y z))
  -/
  induction' x with mx sx
  /-
    case h
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    y z : LocalizedModule S M
    mx : M
    sx : Subtype fun x => Membership.mem S x
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (LocalizedModule.mk mx sx) y) z) (HAdd.hAdd (Locali …
  -/
  induction' y with my sy
  /-
    case h.h
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    z : LocalizedModule S M
    mx : M
    sx : Subtype fun x => Membership.mem S x
    my : M
    sy : Subtype fun x => Membership.mem S x
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (LocalizedModule.mk mx sx) (LocalizedModule.mk my s …
  -/
  induction' z with mz sz
  /-
    case h.h.h
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    mx : M
    sx : Subtype fun x => Membership.mem S x
    my : M
    sy : Subtype fun x => Membership.mem S x
    mz : M
    sz : Subtype fun x => Membership.mem S x
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (LocalizedModule.mk mx sx) (LocalizedModule.mk my s …
  -/
  simp only [mk_add_mk, smul_add]
  /-
    case h.h.h
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    mx : M
    sx : Subtype fun x => Membership.mem S x
    my : M
    sy : Subtype fun x => Membership.mem S x
    mz : M
    sz : Subtype fun x => Membership.mem S x
    ⊢ Eq (LocalizedModule.mk (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul sz (HSMul.hSMul sy …
  -/
  refine mk_eq.mpr ⟨1, ?_⟩
  /-
    case h.h.h
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    mx : M
    sx : Subtype fun x => Membership.mem S x
    my : M
    sy : Subtype fun x => Membership.mem S x
    mz : M
    sz : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul 1 (HSMul.hSMul (HMul.hMul sx (HMul.hMul sy sz)) (HAdd.hAdd ( …
  -/
  rw [one_smul, one_smul]
  /-
    case h.h.h
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    mx : M
    sx : Subtype fun x => Membership.mem S x
    my : M
    sy : Subtype fun x => Membership.mem S x
    mz : M
    sz : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (HMul.hMul sx (HMul.hMul sy sz)) (HAdd.hAdd (HAdd.hAdd (HSMu …
  -/
  congr 1
    /-
      case h.h.h.e_a
      R : Type u
      inst✝² : CommSemiring R
      S : Submonoid R
      M : Type v
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      mx : M
      sx : Subtype fun x => Membership.mem S x
      my : M
      sy : Subtype fun x => Membership.mem S x
      mz : M
      sz : Subtype fun x => Membership.mem S x
      ⊢ Eq (HMul.hMul sx (HMul.hMul sy sz)) (HMul.hMul (HMul.hMul sx sy) sz)
    -/
  · rw [mul_assoc]
    /-
      🎉 no goals
    -/
    /-
      case h.h.h.e_a
      R : Type u
      inst✝² : CommSemiring R
      S : Submonoid R
      M : Type v
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      mx : M
      sx : Subtype fun x => Membership.mem S x
      my : M
      sy : Subtype fun x => Membership.mem S x
      mz : M
      sz : Subtype fun x => Membership.mem S x
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul sz (HSMul.hSMul sy mx)) (HSMul.hSMul s …
    -/
  · rw [eq_comm, mul_comm, add_assoc, mul_smul, mul_smul, ← mul_smul sx sz, mul_comm, mul_smul]
    /-
      🎉 no goals
    -/


private theorem add_comm' (x y : LocalizedModule S M) : x + y = y + x :=
                                                     /-
                                                       R : Type u
                                                       inst✝² : CommSemiring R
                                                       S : Submonoid R
                                                       M : Type v
                                                       inst✝¹ : AddCommMonoid M
                                                       inst✝ : Module R M
                                                       x y : LocalizedModule S M
                                                       m m' : M
                                                       s s' : Subtype fun x => Membership.mem S x
                                                       ⊢ Eq (HAdd.hAdd (LocalizedModule.mk m s) (LocalizedModule.mk m' s')) (HAdd.hAd …
                                                     -/
  LocalizedModule.induction_on₂ (fun m m' s s' => by rw [mk_add_mk, mk_add_mk, add_comm, mul_comm])
                                                     /-
                                                       🎉 no goals
                                                     -/
    x y


private theorem zero_add' (x : LocalizedModule S M) : 0 + x = x :=
  induction_on
    (fun m s => by
      /-
        R : Type u
        inst✝² : CommSemiring R
        S : Submonoid R
        M : Type v
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        x : LocalizedModule S M
        m : M
        s : Subtype fun x => Membership.mem S x
        ⊢ Eq (HAdd.hAdd 0 (LocalizedModule.mk m s)) (LocalizedModule.mk m s)
      -/
      rw [← zero_mk s, mk_add_mk, smul_zero, zero_add, mk_eq]
      /-
        R : Type u
        inst✝² : CommSemiring R
        S : Submonoid R
        M : Type v
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        x : LocalizedModule S M
        m : M
        s : Subtype fun x => Membership.mem S x
        ⊢ Exists fun u => Eq (HSMul.hSMul u (HSMul.hSMul s (HSMul.hSMul s m))) (HSMul. …
      -/
      exact ⟨1, by rw [one_smul, mul_smul, one_smul]⟩)
      /-
        🎉 no goals
      -/
    x


private theorem add_zero' (x : LocalizedModule S M) : x + 0 = x :=
  induction_on
    (fun m s => by
      /-
        R : Type u
        inst✝² : CommSemiring R
        S : Submonoid R
        M : Type v
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        x : LocalizedModule S M
        m : M
        s : Subtype fun x => Membership.mem S x
        ⊢ Eq (HAdd.hAdd (LocalizedModule.mk m s) 0) (LocalizedModule.mk m s)
      -/
      rw [← zero_mk s, mk_add_mk, smul_zero, add_zero, mk_eq]
      /-
        R : Type u
        inst✝² : CommSemiring R
        S : Submonoid R
        M : Type v
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        x : LocalizedModule S M
        m : M
        s : Subtype fun x => Membership.mem S x
        ⊢ Exists fun u => Eq (HSMul.hSMul u (HSMul.hSMul s (HSMul.hSMul s m))) (HSMul. …
      -/
      exact ⟨1, by rw [one_smul, mul_smul, one_smul]⟩)
      /-
        🎉 no goals
      -/
    x


instance hasNatSMul : SMul ℕ (LocalizedModule S M) where smul n := nsmulRec n


private theorem nsmul_zero' (x : LocalizedModule S M) : (0 : ℕ) • x = 0 :=
  LocalizedModule.induction_on (fun _ _ => rfl) x


private theorem nsmul_succ' (n : ℕ) (x : LocalizedModule S M) : n.succ • x = n • x + x :=
  LocalizedModule.induction_on (fun _ _ => rfl) x


instance : AddCommMonoid (LocalizedModule S M) where
  add := (· + ·)
  add_assoc := add_assoc'
  zero := 0
  zero_add := zero_add'
  add_zero := add_zero'
  nsmul := (· • ·)
  nsmul_zero := nsmul_zero'
  nsmul_succ := nsmul_succ'
  add_comm := add_comm'


instance {M : Type*} [AddCommGroup M] [Module R M] : Neg (LocalizedModule S M) where
  neg p :=
    liftOn p (fun x => LocalizedModule.mk (-x.1) x.2) fun ⟨m1, s1⟩ ⟨m2, s2⟩ ⟨u, hu⟩ => by
      /-
        R : Type u
        inst✝⁷ : CommSemiring R
        S : Submonoid R
        M✝ : Type v
        inst✝⁶ : AddCommMonoid M✝
        inst✝⁵ : Module R M✝
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S T
        M : Type u_2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        p : LocalizedModule S M
        x✝² x✝¹ : Prod M (Subtype fun x => Membership.mem S x)
        m1 : M
        s1 : Subtype fun x => Membership.mem S x
        m2 : M
        s2 : Subtype fun x => Membership.mem S x
        x✝ : HasEquiv.Equiv { fst := m1, snd := s1 } { fst := m2, snd := s2 }
        u : Subtype fun x => Membership.mem S x
        hu : Eq (HSMul.hSMul u (HSMul.hSMul { fst := m2, snd := s2 }.2 { fst := m1, sn …
        ⊢ Eq ((fun x => LocalizedModule.mk (Neg.neg x.1) x.2) { fst := m1, snd := s1 } …
      -/
      rw [mk_eq]
      /-
        R : Type u
        inst✝⁷ : CommSemiring R
        S : Submonoid R
        M✝ : Type v
        inst✝⁶ : AddCommMonoid M✝
        inst✝⁵ : Module R M✝
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S T
        M : Type u_2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        p : LocalizedModule S M
        x✝² x✝¹ : Prod M (Subtype fun x => Membership.mem S x)
        m1 : M
        s1 : Subtype fun x => Membership.mem S x
        m2 : M
        s2 : Subtype fun x => Membership.mem S x
        x✝ : HasEquiv.Equiv { fst := m1, snd := s1 } { fst := m2, snd := s2 }
        u : Subtype fun x => Membership.mem S x
        hu : Eq (HSMul.hSMul u (HSMul.hSMul { fst := m2, snd := s2 }.2 { fst := m1, sn …
        ⊢ Exists fun u => Eq (HSMul.hSMul u (HSMul.hSMul { fst := m2, snd := s2 }.2 (N …
      -/
      exact ⟨u, by simpa⟩
      /-
        🎉 no goals
      -/


instance {M : Type*} [AddCommGroup M] [Module R M] : AddCommGroup (LocalizedModule S M) :=
                                                /-
                                                  R : Type u
                                                  inst✝⁷ : CommSemiring R
                                                  S : Submonoid R
                                                  M✝ : Type v
                                                  inst✝⁶ : AddCommMonoid M✝
                                                  inst✝⁵ : Module R M✝
                                                  T : Type u_1
                                                  inst✝⁴ : CommSemiring T
                                                  inst✝³ : Algebra R T
                                                  inst✝² : IsLocalization S T
                                                  M : Type u_2
                                                  inst✝¹ : AddCommGroup M
                                                  inst✝ : Module R M
                                                  ⊢ AddCommMonoid (LocalizedModule S M)
                                                -/
  { show AddCommMonoid (LocalizedModule S M) by infer_instance with
                                                /-
                                                  🎉 no goals
                                                -/
    neg_add_cancel := by
      /-
        R : Type u
        inst✝⁷ : CommSemiring R
        S : Submonoid R
        M✝ : Type v
        inst✝⁶ : AddCommMonoid M✝
        inst✝⁵ : Module R M✝
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S T
        M : Type u_2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        ⊢ ∀ (a : LocalizedModule S M), Eq (HAdd.hAdd (Neg.neg a) a) 0
      -/
      rintro ⟨m, s⟩
      change
        (liftOn (mk m s) (fun x => mk (-x.1) x.2) fun ⟨m1, s1⟩ ⟨m2, s2⟩ ⟨u, hu⟩ => by
              rw [mk_eq]
              exact ⟨u, by simpa⟩) +
            mk m s =
          0
      /-
        case mk.mk
        R : Type u
        inst✝⁷ : CommSemiring R
        S : Submonoid R
        M✝ : Type v
        inst✝⁶ : AddCommMonoid M✝
        inst✝⁵ : Module R M✝
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S T
        M : Type u_2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        a✝ : LocalizedModule S M
        m : M
        s : Subtype fun x => Membership.mem S x
        ⊢ Eq (HAdd.hAdd ((LocalizedModule.mk m s).liftOn (fun x => LocalizedModule.mk  …
      -/
      rw [liftOn_mk, mk_add_mk]
      /-
        case mk.mk
        R : Type u
        inst✝⁷ : CommSemiring R
        S : Submonoid R
        M✝ : Type v
        inst✝⁶ : AddCommMonoid M✝
        inst✝⁵ : Module R M✝
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S T
        M : Type u_2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        a✝ : LocalizedModule S M
        m : M
        s : Subtype fun x => Membership.mem S x
        ⊢ Eq (LocalizedModule.mk (HAdd.hAdd (HSMul.hSMul s (Neg.neg { fst := m, snd := …
      -/
      simp
      /-
        🎉 no goals
      -/
    -- TODO: fix the diamond
    zsmul := zsmulRec }


theorem mk_neg {M : Type*} [AddCommGroup M] [Module R M] {m : M} {s : S} : mk (-m) s = -mk m s :=
  rfl


instance {A : Type*} [Semiring A] [Algebra R A] {S : Submonoid R} :
    Monoid (LocalizedModule S A) :=
  { mul := fun m₁ m₂ =>
      liftOn₂ m₁ m₂ (fun x₁ x₂ => LocalizedModule.mk (x₁.1 * x₂.1) (x₁.2 * x₂.2))
        (by
          /-
            R : Type u
            inst✝⁷ : CommSemiring R
            S✝ : Submonoid R
            M : Type v
            inst✝⁶ : AddCommMonoid M
            inst✝⁵ : Module R M
            T : Type u_1
            inst✝⁴ : CommSemiring T
            inst✝³ : Algebra R T
            inst✝² : IsLocalization S✝ T
            A : Type u_2
            inst✝¹ : Semiring A
            inst✝ : Algebra R A
            S : Submonoid R
            m₁ m₂ : LocalizedModule S A
            ⊢ ∀ (p q p' q' : Prod A (Subtype fun x => Membership.mem S x)), HasEquiv.Equiv …
          -/
          rintro ⟨a₁, s₁⟩ ⟨a₂, s₂⟩ ⟨b₁, t₁⟩ ⟨b₂, t₂⟩ ⟨u₁, e₁⟩ ⟨u₂, e₂⟩
          /-
            case mk.mk.mk.mk.intro.intro
            R : Type u
            inst✝⁷ : CommSemiring R
            S✝ : Submonoid R
            M : Type v
            inst✝⁶ : AddCommMonoid M
            inst✝⁵ : Module R M
            T : Type u_1
            inst✝⁴ : CommSemiring T
            inst✝³ : Algebra R T
            inst✝² : IsLocalization S✝ T
            A : Type u_2
            inst✝¹ : Semiring A
            inst✝ : Algebra R A
            S : Submonoid R
            m₁ m₂ : LocalizedModule S A
            a₁ : A
            s₁ : Subtype fun x => Membership.mem S x
            a₂ : A
            s₂ : Subtype fun x => Membership.mem S x
            b₁ : A
            t₁ : Subtype fun x => Membership.mem S x
            b₂ : A
            t₂ u₁ : Subtype fun x => Membership.mem S x
            e₁ : Eq (HSMul.hSMul u₁ (HSMul.hSMul { fst := b₁, snd := t₁ }.2 { fst := a₁, s …
            u₂ : Subtype fun x => Membership.mem S x
            e₂ : Eq (HSMul.hSMul u₂ (HSMul.hSMul { fst := b₂, snd := t₂ }.2 { fst := a₂, s …
            ⊢ Eq ((fun x₁ x₂ => LocalizedModule.mk (HMul.hMul x₁.1 x₂.1) (HMul.hMul x₁.2 x …
          -/
          rw [mk_eq]
          /-
            case mk.mk.mk.mk.intro.intro
            R : Type u
            inst✝⁷ : CommSemiring R
            S✝ : Submonoid R
            M : Type v
            inst✝⁶ : AddCommMonoid M
            inst✝⁵ : Module R M
            T : Type u_1
            inst✝⁴ : CommSemiring T
            inst✝³ : Algebra R T
            inst✝² : IsLocalization S✝ T
            A : Type u_2
            inst✝¹ : Semiring A
            inst✝ : Algebra R A
            S : Submonoid R
            m₁ m₂ : LocalizedModule S A
            a₁ : A
            s₁ : Subtype fun x => Membership.mem S x
            a₂ : A
            s₂ : Subtype fun x => Membership.mem S x
            b₁ : A
            t₁ : Subtype fun x => Membership.mem S x
            b₂ : A
            t₂ u₁ : Subtype fun x => Membership.mem S x
            e₁ : Eq (HSMul.hSMul u₁ (HSMul.hSMul { fst := b₁, snd := t₁ }.2 { fst := a₁, s …
            u₂ : Subtype fun x => Membership.mem S x
            e₂ : Eq (HSMul.hSMul u₂ (HSMul.hSMul { fst := b₂, snd := t₂ }.2 { fst := a₂, s …
            ⊢ Exists fun u => Eq (HSMul.hSMul u (HSMul.hSMul (HMul.hMul { fst := b₁, snd : …
          -/
          use u₁ * u₂
          /-
            case h
            R : Type u
            inst✝⁷ : CommSemiring R
            S✝ : Submonoid R
            M : Type v
            inst✝⁶ : AddCommMonoid M
            inst✝⁵ : Module R M
            T : Type u_1
            inst✝⁴ : CommSemiring T
            inst✝³ : Algebra R T
            inst✝² : IsLocalization S✝ T
            A : Type u_2
            inst✝¹ : Semiring A
            inst✝ : Algebra R A
            S : Submonoid R
            m₁ m₂ : LocalizedModule S A
            a₁ : A
            s₁ : Subtype fun x => Membership.mem S x
            a₂ : A
            s₂ : Subtype fun x => Membership.mem S x
            b₁ : A
            t₁ : Subtype fun x => Membership.mem S x
            b₂ : A
            t₂ u₁ : Subtype fun x => Membership.mem S x
            e₁ : Eq (HSMul.hSMul u₁ (HSMul.hSMul { fst := b₁, snd := t₁ }.2 { fst := a₁, s …
            u₂ : Subtype fun x => Membership.mem S x
            e₂ : Eq (HSMul.hSMul u₂ (HSMul.hSMul { fst := b₂, snd := t₂ }.2 { fst := a₂, s …
            ⊢ Eq (HSMul.hSMul (HMul.hMul u₁ u₂) (HSMul.hSMul (HMul.hMul { fst := b₁, snd : …
          -/
          dsimp only at e₁ e₂ ⊢
          /-
            case h
            R : Type u
            inst✝⁷ : CommSemiring R
            S✝ : Submonoid R
            M : Type v
            inst✝⁶ : AddCommMonoid M
            inst✝⁵ : Module R M
            T : Type u_1
            inst✝⁴ : CommSemiring T
            inst✝³ : Algebra R T
            inst✝² : IsLocalization S✝ T
            A : Type u_2
            inst✝¹ : Semiring A
            inst✝ : Algebra R A
            S : Submonoid R
            m₁ m₂ : LocalizedModule S A
            a₁ : A
            s₁ : Subtype fun x => Membership.mem S x
            a₂ : A
            s₂ : Subtype fun x => Membership.mem S x
            b₁ : A
            t₁ : Subtype fun x => Membership.mem S x
            b₂ : A
            t₂ u₁ : Subtype fun x => Membership.mem S x
            e₁ : Eq (HSMul.hSMul u₁ (HSMul.hSMul t₁ a₁)) (HSMul.hSMul u₁ (HSMul.hSMul s₁ b …
            u₂ : Subtype fun x => Membership.mem S x
            e₂ : Eq (HSMul.hSMul u₂ (HSMul.hSMul t₂ a₂)) (HSMul.hSMul u₂ (HSMul.hSMul s₂ b …
            ⊢ Eq (HSMul.hSMul (HMul.hMul u₁ u₂) (HSMul.hSMul (HMul.hMul t₁ t₂) (HMul.hMul  …
          -/
          rw [eq_comm]
          /-
            case h
            R : Type u
            inst✝⁷ : CommSemiring R
            S✝ : Submonoid R
            M : Type v
            inst✝⁶ : AddCommMonoid M
            inst✝⁵ : Module R M
            T : Type u_1
            inst✝⁴ : CommSemiring T
            inst✝³ : Algebra R T
            inst✝² : IsLocalization S✝ T
            A : Type u_2
            inst✝¹ : Semiring A
            inst✝ : Algebra R A
            S : Submonoid R
            m₁ m₂ : LocalizedModule S A
            a₁ : A
            s₁ : Subtype fun x => Membership.mem S x
            a₂ : A
            s₂ : Subtype fun x => Membership.mem S x
            b₁ : A
            t₁ : Subtype fun x => Membership.mem S x
            b₂ : A
            t₂ u₁ : Subtype fun x => Membership.mem S x
            e₁ : Eq (HSMul.hSMul u₁ (HSMul.hSMul t₁ a₁)) (HSMul.hSMul u₁ (HSMul.hSMul s₁ b …
            u₂ : Subtype fun x => Membership.mem S x
            e₂ : Eq (HSMul.hSMul u₂ (HSMul.hSMul t₂ a₂)) (HSMul.hSMul u₂ (HSMul.hSMul s₂ b …
            ⊢ Eq (HSMul.hSMul (HMul.hMul u₁ u₂) (HSMul.hSMul (HMul.hMul s₁ s₂) (HMul.hMul  …
          -/
          trans (u₁ • t₁ • a₁) • u₂ • t₂ • a₂
          /-
            R : Type u
            inst✝⁷ : CommSemiring R
            S✝ : Submonoid R
            M : Type v
            inst✝⁶ : AddCommMonoid M
            inst✝⁵ : Module R M
            T : Type u_1
            inst✝⁴ : CommSemiring T
            inst✝³ : Algebra R T
            inst✝² : IsLocalization S✝ T
            A : Type u_2
            inst✝¹ : Semiring A
            inst✝ : Algebra R A
            S : Submonoid R
            m₁ m₂ : LocalizedModule S A
            a₁ : A
            s₁ : Subtype fun x => Membership.mem S x
            a₂ : A
            s₂ : Subtype fun x => Membership.mem S x
            b₁ : A
            t₁ : Subtype fun x => Membership.mem S x
            b₂ : A
            t₂ u₁ : Subtype fun x => Membership.mem S x
            e₁ : Eq (HSMul.hSMul u₁ (HSMul.hSMul t₁ a₁)) (HSMul.hSMul u₁ (HSMul.hSMul s₁ b …
            u₂ : Subtype fun x => Membership.mem S x
            e₂ : Eq (HSMul.hSMul u₂ (HSMul.hSMul t₂ a₂)) (HSMul.hSMul u₂ (HSMul.hSMul s₂ b …
            ⊢ Eq (HSMul.hSMul (HMul.hMul u₁ u₂) (HSMul.hSMul (HMul.hMul s₁ s₂) (HMul.hMul  …
          -/
          on_goal 1 => rw [e₁, e₂]
          /-
            R : Type u
            inst✝⁷ : CommSemiring R
            S✝ : Submonoid R
            M : Type v
            inst✝⁶ : AddCommMonoid M
            inst✝⁵ : Module R M
            T : Type u_1
            inst✝⁴ : CommSemiring T
            inst✝³ : Algebra R T
            inst✝² : IsLocalization S✝ T
            A : Type u_2
            inst✝¹ : Semiring A
            inst✝ : Algebra R A
            S : Submonoid R
            m₁ m₂ : LocalizedModule S A
            a₁ : A
            s₁ : Subtype fun x => Membership.mem S x
            a₂ : A
            s₂ : Subtype fun x => Membership.mem S x
            b₁ : A
            t₁ : Subtype fun x => Membership.mem S x
            b₂ : A
            t₂ u₁ : Subtype fun x => Membership.mem S x
            e₁ : Eq (HSMul.hSMul u₁ (HSMul.hSMul t₁ a₁)) (HSMul.hSMul u₁ (HSMul.hSMul s₁ b …
            u₂ : Subtype fun x => Membership.mem S x
            e₂ : Eq (HSMul.hSMul u₂ (HSMul.hSMul t₂ a₂)) (HSMul.hSMul u₂ (HSMul.hSMul s₂ b …
            ⊢ Eq (HSMul.hSMul (HMul.hMul u₁ u₂) (HSMul.hSMul (HMul.hMul s₁ s₂) (HMul.hMul  …
          -/
          on_goal 2 => rw [eq_comm]
          all_goals
            rw [smul_smul, mul_mul_mul_comm, ← smul_eq_mul, ← smul_eq_mul A, smul_smul_smul_comm,
              mul_smul, mul_smul])
    one := mk 1 (1 : S)
    one_mul := by
      /-
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Submonoid R
        ⊢ ∀ (a : LocalizedModule S A), Eq (HMul.hMul 1 a) a
      -/
      rintro ⟨a, s⟩
      /-
        case mk.mk
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Submonoid R
        a✝ : LocalizedModule S A
        a : A
        s : Subtype fun x => Membership.mem S x
        ⊢ Eq (HMul.hMul 1 (Quot.mk ⇑(LocalizedModule.r.setoid S A) { fst := a, snd :=  …
      -/
      exact mk_eq.mpr ⟨1, by simp only [one_mul, one_smul]⟩
      /-
        🎉 no goals
      -/
    mul_one := by
      /-
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Submonoid R
        ⊢ ∀ (a b c : LocalizedModule S A), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul …
      -/
      /-
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Submonoid R
        ⊢ ∀ (a : LocalizedModule S A), Eq (HMul.hMul a 1) a
      -/
      /-
        case mk.mk.mk.mk.mk.mk
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Submonoid R
        a✝ : LocalizedModule S A
        a₁ : A
        s₁ : Subtype fun x => Membership.mem S x
        b✝ : LocalizedModule S A
        a₂ : A
        s₂ : Subtype fun x => Membership.mem S x
        c✝ : LocalizedModule S A
        a₃ : A
        s₃ : Subtype fun x => Membership.mem S x
        ⊢ Eq (HMul.hMul (HMul.hMul (Quot.mk ⇑(LocalizedModule.r.setoid S A) { fst := a …
      -/
      rintro ⟨a, s⟩
      /-
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Submonoid R
        a✝ : LocalizedModule S A
        a₁ : A
        s₁ : Subtype fun x => Membership.mem S x
        b✝ : LocalizedModule S A
        a₂ : A
        s₂ : Subtype fun x => Membership.mem S x
        c✝ : LocalizedModule S A
        a₃ : A
        s₃ : Subtype fun x => Membership.mem S x
        ⊢ Exists fun u => Eq (HSMul.hSMul u (HSMul.hSMul (HMul.hMul { fst := a₁, snd : …
      -/
      /-
        case mk.mk
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Submonoid R
        a✝ : LocalizedModule S A
        a : A
        s : Subtype fun x => Membership.mem S x
        ⊢ Eq (HMul.hMul (Quot.mk ⇑(LocalizedModule.r.setoid S A) { fst := a, snd := s  …
      -/
      /-
        case h
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Submonoid R
        a✝ : LocalizedModule S A
        a₁ : A
        s₁ : Subtype fun x => Membership.mem S x
        b✝ : LocalizedModule S A
        a₂ : A
        s₂ : Subtype fun x => Membership.mem S x
        c✝ : LocalizedModule S A
        a₃ : A
        s₃ : Subtype fun x => Membership.mem S x
        ⊢ Eq (HSMul.hSMul 1 (HSMul.hSMul (HMul.hMul { fst := a₁, snd := s₁ }.2 { fst : …
      -/
      exact mk_eq.mpr ⟨1, by simp only [mul_one, one_smul]⟩
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
    mul_assoc := by
      rintro ⟨a₁, s₁⟩ ⟨a₂, s₂⟩ ⟨a₃, s₃⟩
      apply mk_eq.mpr _
      use 1
      simp only [one_mul, smul_smul, ← mul_assoc, mul_right_comm] }


instance {A : Type*} [Semiring A] [Algebra R A] {S : Submonoid R} :
    Semiring (LocalizedModule S A) :=
                                                  /-
                                                    R : Type u
                                                    inst✝⁷ : CommSemiring R
                                                    S✝ : Submonoid R
                                                    M : Type v
                                                    inst✝⁶ : AddCommMonoid M
                                                    inst✝⁵ : Module R M
                                                    T : Type u_1
                                                    inst✝⁴ : CommSemiring T
                                                    inst✝³ : Algebra R T
                                                    inst✝² : IsLocalization S✝ T
                                                    A : Type u_2
                                                    inst✝¹ : Semiring A
                                                    inst✝ : Algebra R A
                                                    S : Submonoid R
                                                    ⊢ AddCommMonoid (LocalizedModule S A)
                                                  -/
  { show (AddCommMonoid (LocalizedModule S A)) by infer_instance,
                                                  /-
                                                    🎉 no goals
                                                  -/
                                           /-
                                             R : Type u
                                             inst✝⁷ : CommSemiring R
                                             S✝ : Submonoid R
                                             M : Type v
                                             inst✝⁶ : AddCommMonoid M
                                             inst✝⁵ : Module R M
                                             T : Type u_1
                                             inst✝⁴ : CommSemiring T
                                             inst✝³ : Algebra R T
                                             inst✝² : IsLocalization S✝ T
                                             A : Type u_2
                                             inst✝¹ : Semiring A
                                             inst✝ : Algebra R A
                                             S : Submonoid R
                                             ⊢ Monoid (LocalizedModule S A)
                                           -/
    show (Monoid (LocalizedModule S A)) by infer_instance with
                                           /-
                                             🎉 no goals
                                           -/
    left_distrib := by
      /-
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Submonoid R
        ⊢ ∀ (a b c : LocalizedModule S A), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd …
      -/
      rintro ⟨a₁, s₁⟩ ⟨a₂, s₂⟩ ⟨a₃, s₃⟩
      /-
        case mk.mk.mk.mk.mk.mk
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Submonoid R
        a✝ : LocalizedModule S A
        a₁ : A
        s₁ : Subtype fun x => Membership.mem S x
        b✝ : LocalizedModule S A
        a₂ : A
        s₂ : Subtype fun x => Membership.mem S x
        c✝ : LocalizedModule S A
        a₃ : A
        s₃ : Subtype fun x => Membership.mem S x
        ⊢ Eq (HMul.hMul (Quot.mk ⇑(LocalizedModule.r.setoid S A) { fst := a₁, snd := s …
      -/
      apply mk_eq.mpr _
      /-
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Submonoid R
        a✝ : LocalizedModule S A
        a₁ : A
        s₁ : Subtype fun x => Membership.mem S x
        b✝ : LocalizedModule S A
        a₂ : A
        s₂ : Subtype fun x => Membership.mem S x
        c✝ : LocalizedModule S A
        a₃ : A
        s₃ : Subtype fun x => Membership.mem S x
        ⊢ Exists fun u => Eq (HSMul.hSMul u (HSMul.hSMul (HMul.hMul { fst := HMul.hMul …
      -/
      use 1
      simp only [one_mul, smul_add, mul_add, mul_smul_comm, smul_smul, ← mul_assoc,
        mul_right_comm]
    right_distrib := by
      /-
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Submonoid R
        ⊢ ∀ (a b c : LocalizedModule S A), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd …
      -/
      rintro ⟨a₁, s₁⟩ ⟨a₂, s₂⟩ ⟨a₃, s₃⟩
      /-
        case mk.mk.mk.mk.mk.mk
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Submonoid R
        a✝ : LocalizedModule S A
        a₁ : A
        s₁ : Subtype fun x => Membership.mem S x
        b✝ : LocalizedModule S A
        a₂ : A
        s₂ : Subtype fun x => Membership.mem S x
        c✝ : LocalizedModule S A
        a₃ : A
        s₃ : Subtype fun x => Membership.mem S x
        ⊢ Eq (HMul.hMul (HAdd.hAdd (Quot.mk ⇑(LocalizedModule.r.setoid S A) { fst := a …
      -/
      apply mk_eq.mpr _
      /-
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Submonoid R
        a✝ : LocalizedModule S A
        a₁ : A
        s₁ : Subtype fun x => Membership.mem S x
        b✝ : LocalizedModule S A
        a₂ : A
        s₂ : Subtype fun x => Membership.mem S x
        c✝ : LocalizedModule S A
        a₃ : A
        s₃ : Subtype fun x => Membership.mem S x
        ⊢ Exists fun u => Eq (HSMul.hSMul u (HSMul.hSMul (HMul.hMul { fst := HMul.hMul …
      -/
      use 1
      simp only [one_mul, smul_add, add_mul, smul_smul, ← mul_assoc, smul_mul_assoc,
        mul_right_comm]
    zero_mul := by
      /-
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Submonoid R
        ⊢ ∀ (a : LocalizedModule S A), Eq (HMul.hMul 0 a) 0
      -/
      rintro ⟨a, s⟩
      /-
        case mk.mk
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Submonoid R
        a✝ : LocalizedModule S A
        a : A
        s : Subtype fun x => Membership.mem S x
        ⊢ Eq (HMul.hMul 0 (Quot.mk ⇑(LocalizedModule.r.setoid S A) { fst := a, snd :=  …
      -/
      exact mk_eq.mpr ⟨1, by simp only [zero_mul, smul_zero]⟩
      /-
        🎉 no goals
      -/
    mul_zero := by
      /-
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Submonoid R
        ⊢ ∀ (a : LocalizedModule S A), Eq (HMul.hMul a 0) 0
      -/
      rintro ⟨a, s⟩
      /-
        case mk.mk
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Submonoid R
        a✝ : LocalizedModule S A
        a : A
        s : Subtype fun x => Membership.mem S x
        ⊢ Eq (HMul.hMul (Quot.mk ⇑(LocalizedModule.r.setoid S A) { fst := a, snd := s  …
      -/
      exact mk_eq.mpr ⟨1, by simp only [mul_zero, smul_zero]⟩ }
      /-
        🎉 no goals
      -/


instance {A : Type*} [CommSemiring A] [Algebra R A] {S : Submonoid R} :
    CommSemiring (LocalizedModule S A) :=
                                           /-
                                             R : Type u
                                             inst✝⁷ : CommSemiring R
                                             S✝ : Submonoid R
                                             M : Type v
                                             inst✝⁶ : AddCommMonoid M
                                             inst✝⁵ : Module R M
                                             T : Type u_1
                                             inst✝⁴ : CommSemiring T
                                             inst✝³ : Algebra R T
                                             inst✝² : IsLocalization S✝ T
                                             A : Type u_2
                                             inst✝¹ : CommSemiring A
                                             inst✝ : Algebra R A
                                             S : Submonoid R
                                             ⊢ Semiring (LocalizedModule S A)
                                           -/
  { show Semiring (LocalizedModule S A) by infer_instance with
                                           /-
                                             🎉 no goals
                                           -/
    mul_comm := by
      /-
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : CommSemiring A
        inst✝ : Algebra R A
        S : Submonoid R
        ⊢ ∀ (a b : LocalizedModule S A), Eq (HMul.hMul a b) (HMul.hMul b a)
      -/
      rintro ⟨a₁, s₁⟩ ⟨a₂, s₂⟩
      /-
        case mk.mk.mk.mk
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : CommSemiring A
        inst✝ : Algebra R A
        S : Submonoid R
        a✝ : LocalizedModule S A
        a₁ : A
        s₁ : Subtype fun x => Membership.mem S x
        b✝ : LocalizedModule S A
        a₂ : A
        s₂ : Subtype fun x => Membership.mem S x
        ⊢ Eq (HMul.hMul (Quot.mk ⇑(LocalizedModule.r.setoid S A) { fst := a₁, snd := s …
      -/
      exact mk_eq.mpr ⟨1, by simp only [one_smul, mul_comm]⟩ }
      /-
        🎉 no goals
      -/


instance {A : Type*} [Ring A] [Algebra R A] {S : Submonoid R} :
    Ring (LocalizedModule S A) :=
  { inferInstanceAs (AddCommGroup (LocalizedModule S A)),
    inferInstanceAs (Semiring (LocalizedModule S A)) with }


instance {A : Type*} [CommRing A] [Algebra R A] {S : Submonoid R} :
    CommRing (LocalizedModule S A) :=
                                         /-
                                           R : Type u
                                           inst✝⁷ : CommSemiring R
                                           S✝ : Submonoid R
                                           M : Type v
                                           inst✝⁶ : AddCommMonoid M
                                           inst✝⁵ : Module R M
                                           T : Type u_1
                                           inst✝⁴ : CommSemiring T
                                           inst✝³ : Algebra R T
                                           inst✝² : IsLocalization S✝ T
                                           A : Type u_2
                                           inst✝¹ : CommRing A
                                           inst✝ : Algebra R A
                                           S : Submonoid R
                                           ⊢ Ring (LocalizedModule S A)
                                         -/
  { show (Ring (LocalizedModule S A)) by infer_instance with
                                         /-
                                           🎉 no goals
                                         -/
    mul_comm := by
      /-
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        S : Submonoid R
        ⊢ ∀ (a b : LocalizedModule S A), Eq (HMul.hMul a b) (HMul.hMul b a)
      -/
      rintro ⟨a₁, s₁⟩ ⟨a₂, s₂⟩
      /-
        case mk.mk.mk.mk
        R : Type u
        inst✝⁷ : CommSemiring R
        S✝ : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S✝ T
        A : Type u_2
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        S : Submonoid R
        a✝ : LocalizedModule S A
        a₁ : A
        s₁ : Subtype fun x => Membership.mem S x
        b✝ : LocalizedModule S A
        a₂ : A
        s₂ : Subtype fun x => Membership.mem S x
        ⊢ Eq (HMul.hMul (Quot.mk ⇑(LocalizedModule.r.setoid S A) { fst := a₁, snd := s …
      -/
      exact mk_eq.mpr ⟨1, by simp only [one_smul, mul_comm]⟩ }
      /-
        🎉 no goals
      -/


theorem mk_mul_mk {A : Type*} [Semiring A] [Algebra R A] {a₁ a₂ : A} {s₁ s₂ : S} :
    mk a₁ s₁ * mk a₂ s₂ = mk (a₁ * a₂) (s₁ * s₂) :=
  rfl


noncomputable instance : SMul T (LocalizedModule S M) where
  smul x p :=
    let a := IsLocalization.sec S x
    liftOn p (fun p ↦ mk (a.1 • p.1) (a.2 * p.2))
      (by
        /-
          R : Type u
          inst✝⁵ : CommSemiring R
          S : Submonoid R
          M : Type v
          inst✝⁴ : AddCommMonoid M
          inst✝³ : Module R M
          T : Type u_1
          inst✝² : CommSemiring T
          inst✝¹ : Algebra R T
          inst✝ : IsLocalization S T
          x : T
          p : LocalizedModule S M
          a : Prod R (Subtype fun x => Membership.mem S x) := IsLocalization.sec S x
          ⊢ ∀ (p p' : Prod M (Subtype fun x => Membership.mem S x)), HasEquiv.Equiv p p' …
        -/
        rintro p p' ⟨s, h⟩
        /-
          case intro
          R : Type u
          inst✝⁵ : CommSemiring R
          S : Submonoid R
          M : Type v
          inst✝⁴ : AddCommMonoid M
          inst✝³ : Module R M
          T : Type u_1
          inst✝² : CommSemiring T
          inst✝¹ : Algebra R T
          inst✝ : IsLocalization S T
          x : T
          p✝ : LocalizedModule S M
          a : Prod R (Subtype fun x => Membership.mem S x) := IsLocalization.sec S x
          p p' : Prod M (Subtype fun x => Membership.mem S x)
          s : Subtype fun x => Membership.mem S x
          h : Eq (HSMul.hSMul s (HSMul.hSMul p'.2 p.1)) (HSMul.hSMul s (HSMul.hSMul p.2  …
          ⊢ Eq ((fun p => LocalizedModule.mk (HSMul.hSMul a.1 p.1) (HMul.hMul a.2 p.2))  …
        -/
        refine mk_eq.mpr ⟨s, ?_⟩
        calc
          _ = a.2 • a.1 • s • p'.2 • p.1 := by
            simp_rw [Submonoid.smul_def, Submonoid.coe_mul, ← mul_smul]; ring_nf
          _ = a.2 • a.1 • s • p.2 • p'.1 := by rw [h]
          _ = s • (a.2 * p.2) • a.1 • p'.1 := by
            simp_rw [Submonoid.smul_def, ← mul_smul, Submonoid.coe_mul]; ring_nf )


theorem smul_def (x : T) (m : M) (s : S) :
    x • mk m s = mk ((IsLocalization.sec S x).1 • m) ((IsLocalization.sec S x).2 * s) := rfl


theorem mk'_smul_mk (r : R) (m : M) (s s' : S) :
    IsLocalization.mk' T r s • mk m s' = mk (r • m) (s * s') := by
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    T : Type u_1
    inst✝² : CommSemiring T
    inst✝¹ : Algebra R T
    inst✝ : IsLocalization S T
    r : R
    m : M
    s s' : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (IsLocalization.mk' T r s) (LocalizedModule.mk m s')) (Local …
  -/
  rw [smul_def, mk_eq]
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    T : Type u_1
    inst✝² : CommSemiring T
    inst✝¹ : Algebra R T
    inst✝ : IsLocalization S T
    r : R
    m : M
    s s' : Subtype fun x => Membership.mem S x
    ⊢ Exists fun u => Eq (HSMul.hSMul u (HSMul.hSMul (HMul.hMul s s') (HSMul.hSMul …
  -/
  obtain ⟨c, hc⟩ := IsLocalization.eq.mp <| IsLocalization.mk'_sec T (IsLocalization.mk' T r s)
  /-
    case intro
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    T : Type u_1
    inst✝² : CommSemiring T
    inst✝¹ : Algebra R T
    inst✝ : IsLocalization S T
    r : R
    m : M
    s s' c : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul (↑c) (HMul.hMul (↑s) (IsLocalization.sec S (IsLocalization. …
    ⊢ Exists fun u => Eq (HSMul.hSMul u (HSMul.hSMul (HMul.hMul s s') (HSMul.hSMul …
  -/
  use c
  simp_rw [← mul_smul, Submonoid.smul_def, Submonoid.coe_mul, ← mul_smul, ← mul_assoc,
    mul_comm _ (s' : R), mul_assoc, hc]


theorem mk_smul_mk (r : R) (m : M) (s t : S) :
    Localization.mk r s • mk m t = mk (r • m) (s * t) := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    r : R
    m : M
    s t : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (Localization.mk r s) (LocalizedModule.mk m t)) (LocalizedMo …
  -/
  rw [Localization.mk_eq_mk']
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    r : R
    m : M
    s t : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (IsLocalization.mk' (Localization S) r s) (LocalizedModule.m …
  -/
  exact mk'_smul_mk ..
  /-
    🎉 no goals
  -/


private theorem one_smul_aux (p : LocalizedModule S M) : (1 : T) • p = p := by
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    T : Type u_1
    inst✝² : CommSemiring T
    inst✝¹ : Algebra R T
    inst✝ : IsLocalization S T
    p : LocalizedModule S M
    ⊢ Eq (HSMul.hSMul 1 p) p
  -/
  induction' p with m s
  /-
    case h
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    T : Type u_1
    inst✝² : CommSemiring T
    inst✝¹ : Algebra R T
    inst✝ : IsLocalization S T
    m : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul 1 (LocalizedModule.mk m s)) (LocalizedModule.mk m s)
  -/
  rw [show (1 : T) = IsLocalization.mk' T (1 : R) (1 : S) by rw [IsLocalization.mk'_one, map_one]]
  /-
    case h
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    T : Type u_1
    inst✝² : CommSemiring T
    inst✝¹ : Algebra R T
    inst✝ : IsLocalization S T
    m : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (IsLocalization.mk' T 1 1) (LocalizedModule.mk m s)) (Locali …
  -/
  rw [mk'_smul_mk, one_smul, one_mul]
  /-
    🎉 no goals
  -/


private theorem mul_smul_aux (x y : T) (p : LocalizedModule S M) :
    (x * y) • p = x • y • p := by
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    T : Type u_1
    inst✝² : CommSemiring T
    inst✝¹ : Algebra R T
    inst✝ : IsLocalization S T
    x y : T
    p : LocalizedModule S M
    ⊢ Eq (HSMul.hSMul (HMul.hMul x y) p) (HSMul.hSMul x (HSMul.hSMul y p))
  -/
  induction' p with m s
  /-
    case h
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    T : Type u_1
    inst✝² : CommSemiring T
    inst✝¹ : Algebra R T
    inst✝ : IsLocalization S T
    x y : T
    m : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (HMul.hMul x y) (LocalizedModule.mk m s)) (HSMul.hSMul x (HS …
  -/
  rw [← IsLocalization.mk'_sec (M := S) T x, ← IsLocalization.mk'_sec (M := S) T y]
  /-
    case h
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    T : Type u_1
    inst✝² : CommSemiring T
    inst✝¹ : Algebra R T
    inst✝ : IsLocalization S T
    x y : T
    m : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (HMul.hMul (IsLocalization.mk' T (IsLocalization.sec S x).1  …
  -/
  simp_rw [← IsLocalization.mk'_mul, mk'_smul_mk, ← mul_smul, mul_assoc]
  /-
    🎉 no goals
  -/


private theorem smul_add_aux (x : T) (p q : LocalizedModule S M) :
    x • (p + q) = x • p + x • q := by
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    T : Type u_1
    inst✝² : CommSemiring T
    inst✝¹ : Algebra R T
    inst✝ : IsLocalization S T
    x : T
    p q : LocalizedModule S M
    ⊢ Eq (HSMul.hSMul x (HAdd.hAdd p q)) (HAdd.hAdd (HSMul.hSMul x p) (HSMul.hSMul …
  -/
  induction' p with m s
  /-
    case h
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    T : Type u_1
    inst✝² : CommSemiring T
    inst✝¹ : Algebra R T
    inst✝ : IsLocalization S T
    x : T
    q : LocalizedModule S M
    m : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul x (HAdd.hAdd (LocalizedModule.mk m s) q)) (HAdd.hAdd (HSMul. …
  -/
  induction' q with n t
  /-
    case h.h
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    T : Type u_1
    inst✝² : CommSemiring T
    inst✝¹ : Algebra R T
    inst✝ : IsLocalization S T
    x : T
    m : M
    s : Subtype fun x => Membership.mem S x
    n : M
    t : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul x (HAdd.hAdd (LocalizedModule.mk m s) (LocalizedModule.mk n  …
  -/
  rw [smul_def, smul_def, mk_add_mk, mk_add_mk]
  /-
    case h.h
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    T : Type u_1
    inst✝² : CommSemiring T
    inst✝¹ : Algebra R T
    inst✝ : IsLocalization S T
    x : T
    m : M
    s : Subtype fun x => Membership.mem S x
    n : M
    t : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul x (LocalizedModule.mk (HAdd.hAdd (HSMul.hSMul t m) (HSMul.hS …
  -/
  rw [show x • _ =  IsLocalization.mk' T _ _ • _ by rw [IsLocalization.mk'_sec (M := S) T]]
  /-
    case h.h
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    T : Type u_1
    inst✝² : CommSemiring T
    inst✝¹ : Algebra R T
    inst✝ : IsLocalization S T
    x : T
    m : M
    s : Subtype fun x => Membership.mem S x
    n : M
    t : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (IsLocalization.mk' T (IsLocalization.sec S x).1 (IsLocaliza …
  -/
  rw [← IsLocalization.mk'_cancel _ _ (IsLocalization.sec S x).2, mk'_smul_mk]
  /-
    case h.h
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    T : Type u_1
    inst✝² : CommSemiring T
    inst✝¹ : Algebra R T
    inst✝ : IsLocalization S T
    x : T
    m : M
    s : Subtype fun x => Membership.mem S x
    n : M
    t : Subtype fun x => Membership.mem S x
    ⊢ Eq (LocalizedModule.mk (HSMul.hSMul (HMul.hMul (IsLocalization.sec S x).1 ↑( …
  -/
  congr 1
    /-
      case h.h.e_m
      R : Type u
      inst✝⁵ : CommSemiring R
      S : Submonoid R
      M : Type v
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      T : Type u_1
      inst✝² : CommSemiring T
      inst✝¹ : Algebra R T
      inst✝ : IsLocalization S T
      x : T
      m : M
      s : Subtype fun x => Membership.mem S x
      n : M
      t : Subtype fun x => Membership.mem S x
      ⊢ Eq (HSMul.hSMul (HMul.hMul (IsLocalization.sec S x).1 ↑(IsLocalization.sec S …
    -/
  · simp only [Submonoid.smul_def, smul_add, ← mul_smul, Submonoid.coe_mul]; ring_nf
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
    /-
      case h.h.e_s
      R : Type u
      inst✝⁵ : CommSemiring R
      S : Submonoid R
      M : Type v
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      T : Type u_1
      inst✝² : CommSemiring T
      inst✝¹ : Algebra R T
      inst✝ : IsLocalization S T
      x : T
      m : M
      s : Subtype fun x => Membership.mem S x
      n : M
      t : Subtype fun x => Membership.mem S x
      ⊢ Eq (HMul.hMul (HMul.hMul (IsLocalization.sec S x).2 (IsLocalization.sec S x) …
    -/
  · rw [mul_mul_mul_comm] -- ring does not work here
    /-
      🎉 no goals
    -/


private theorem smul_zero_aux (x : T) : x • (0 : LocalizedModule S M) = 0 := by
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    T : Type u_1
    inst✝² : CommSemiring T
    inst✝¹ : Algebra R T
    inst✝ : IsLocalization S T
    x : T
    ⊢ Eq (HSMul.hSMul x 0) 0
  -/
  erw [smul_def, smul_zero, zero_mk]
  /-
    🎉 no goals
  -/


private theorem add_smul_aux (x y : T) (p : LocalizedModule S M) :
    (x + y) • p = x • p + y • p := by
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    T : Type u_1
    inst✝² : CommSemiring T
    inst✝¹ : Algebra R T
    inst✝ : IsLocalization S T
    x y : T
    p : LocalizedModule S M
    ⊢ Eq (HSMul.hSMul (HAdd.hAdd x y) p) (HAdd.hAdd (HSMul.hSMul x p) (HSMul.hSMul …
  -/
  induction' p with m s
  rw [smul_def T x, smul_def T y, mk_add_mk, show (x + y) • _ =  IsLocalization.mk' T _ _ • _ by
    rw [← IsLocalization.mk'_sec (M := S) T x, ← IsLocalization.mk'_sec (M := S) T y,
      ← IsLocalization.mk'_add, IsLocalization.mk'_cancel _ _ s], mk'_smul_mk, ← smul_assoc,
    ← smul_assoc, ← add_smul]
  /-
    case h
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    T : Type u_1
    inst✝² : CommSemiring T
    inst✝¹ : Algebra R T
    inst✝ : IsLocalization S T
    x y : T
    m : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (LocalizedModule.mk (HSMul.hSMul (HMul.hMul (HAdd.hAdd (HMul.hMul (IsLoca …
  -/
  congr 1
    /-
      case h.e_m
      R : Type u
      inst✝⁵ : CommSemiring R
      S : Submonoid R
      M : Type v
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      T : Type u_1
      inst✝² : CommSemiring T
      inst✝¹ : Algebra R T
      inst✝ : IsLocalization S T
      x y : T
      m : M
      s : Subtype fun x => Membership.mem S x
      ⊢ Eq (HSMul.hSMul (HMul.hMul (HAdd.hAdd (HMul.hMul (IsLocalization.sec S x).1  …
    -/
  · simp only [Submonoid.smul_def, Submonoid.coe_mul, smul_eq_mul]; ring_nf
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    /-
      case h.e_s
      R : Type u
      inst✝⁵ : CommSemiring R
      S : Submonoid R
      M : Type v
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      T : Type u_1
      inst✝² : CommSemiring T
      inst✝¹ : Algebra R T
      inst✝ : IsLocalization S T
      x y : T
      m : M
      s : Subtype fun x => Membership.mem S x
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (IsLocalization.sec S x).2 (IsLocalizati …
    -/
  · rw [mul_mul_mul_comm, mul_assoc] -- ring does not work here
    /-
      🎉 no goals
    -/


private theorem zero_smul_aux (p : LocalizedModule S M) : (0 : T) • p = 0 := by
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    T : Type u_1
    inst✝² : CommSemiring T
    inst✝¹ : Algebra R T
    inst✝ : IsLocalization S T
    p : LocalizedModule S M
    ⊢ Eq (HSMul.hSMul 0 p) 0
  -/
  induction' p with m s
  rw [show (0 : T) = IsLocalization.mk' T (0 : R) (1 : S) by rw [IsLocalization.mk'_zero],
    mk'_smul_mk, zero_smul, zero_mk]


noncomputable instance isModule : Module T (LocalizedModule S M) where
  smul := (· • ·)
  one_smul := one_smul_aux
  mul_smul := mul_smul_aux
  smul_add := smul_add_aux
  smul_zero := smul_zero_aux
  add_smul := add_smul_aux
  zero_smul := zero_smul_aux


@[simp]
theorem mk_cancel_common_left (s' s : S) (m : M) : mk (s' • m) (s' * s) = mk m s :=
  mk_eq.mpr
    ⟨1, by
      /-
        R : Type u
        inst✝² : CommSemiring R
        S : Submonoid R
        M : Type v
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        s' s : Subtype fun x => Membership.mem S x
        m : M
        ⊢ Eq (HSMul.hSMul 1 (HSMul.hSMul s (HSMul.hSMul s' m))) (HSMul.hSMul 1 (HSMul. …
      -/
      simp only [mul_smul, one_smul]
      /-
        R : Type u
        inst✝² : CommSemiring R
        S : Submonoid R
        M : Type v
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        s' s : Subtype fun x => Membership.mem S x
        m : M
        ⊢ Eq (HSMul.hSMul s (HSMul.hSMul s' m)) (HSMul.hSMul s' (HSMul.hSMul s m))
      -/
      rw [smul_comm]⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem mk_cancel (s : S) (m : M) : mk (s • m) s = mk m 1 :=
                   /-
                     R : Type u
                     inst✝² : CommSemiring R
                     S : Submonoid R
                     M : Type v
                     inst✝¹ : AddCommMonoid M
                     inst✝ : Module R M
                     s : Subtype fun x => Membership.mem S x
                     m : M
                     ⊢ Eq (HSMul.hSMul 1 (HSMul.hSMul 1 (HSMul.hSMul s m))) (HSMul.hSMul 1 (HSMul.h …
                   -/
  mk_eq.mpr ⟨1, by simp⟩
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem mk_cancel_common_right (s s' : S) (m : M) : mk (s' • m) (s * s') = mk m s :=
                   /-
                     R : Type u
                     inst✝² : CommSemiring R
                     S : Submonoid R
                     M : Type v
                     inst✝¹ : AddCommMonoid M
                     inst✝ : Module R M
                     s s' : Subtype fun x => Membership.mem S x
                     m : M
                     ⊢ Eq (HSMul.hSMul 1 (HSMul.hSMul s (HSMul.hSMul s' m))) (HSMul.hSMul 1 (HSMul. …
                   -/
  mk_eq.mpr ⟨1, by simp [mul_smul]⟩
                   /-
                     🎉 no goals
                   -/


noncomputable instance isModule' : Module R (LocalizedModule S M) :=
  { Module.compHom (LocalizedModule S M) <| algebraMap R (Localization S) with }


theorem smul'_mk (r : R) (s : S) (m : M) : r • mk m s = mk (r • m) s := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    r : R
    s : Subtype fun x => Membership.mem S x
    m : M
    ⊢ Eq (HSMul.hSMul r (LocalizedModule.mk m s)) (LocalizedModule.mk (HSMul.hSMul …
  -/
  erw [mk_smul_mk r m 1 s, one_mul]
  /-
    🎉 no goals
  -/


lemma smul_eq_iff_of_mem
    (r : R) (hr : r ∈ S) (x y : LocalizedModule S M) :
    r • x = y ↔ x = Localization.mk 1 ⟨r, hr⟩ • y := by
  induction x using induction_on with
  | h m s =>
    induction y using induction_on with
    | h n t =>
      rw [smul'_mk, mk_smul_mk, one_smul, mk_eq, mk_eq]
      simp only [Subtype.exists, Submonoid.mk_smul, exists_prop]
      fconstructor
      · rintro ⟨a, ha, eq1⟩
        refine ⟨a, ha, ?_⟩
        rw [mul_smul, ← eq1, Submonoid.mk_smul, smul_comm r t]
      · rintro ⟨a, ha, eq1⟩
        refine ⟨a, ha, ?_⟩
        rw [← eq1, mul_comm, mul_smul, Submonoid.mk_smul, Submonoid.smul_def, Submonoid.mk_smul]


lemma eq_zero_of_smul_eq_zero
    (r : R) (hr : r ∈ S) (x : LocalizedModule S M) (hx : r • x = 0) : x = 0 := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    r : R
    hr : Membership.mem S r
    x : LocalizedModule S M
    hx : Eq (HSMul.hSMul r x) 0
    ⊢ Eq x 0
  -/
  rw [smul_eq_iff_of_mem (hr := hr)] at hx
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    r : R
    hr : Membership.mem S r
    x : LocalizedModule S M
    hx : Eq x (HSMul.hSMul (Localization.mk 1 ⟨r, hr⟩) 0)
    ⊢ Eq x 0
  -/
  rw [hx, smul_zero]
  /-
    🎉 no goals
  -/


theorem smul'_mul {A : Type*} [Semiring A] [Algebra R A] (x : T) (p₁ p₂ : LocalizedModule S A) :
    x • p₁ * p₂ = x • (p₁ * p₂) := by
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    T : Type u_1
    inst✝⁴ : CommSemiring T
    inst✝³ : Algebra R T
    inst✝² : IsLocalization S T
    A : Type u_2
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : T
    p₁ p₂ : LocalizedModule S A
    ⊢ Eq (HMul.hMul (HSMul.hSMul x p₁) p₂) (HSMul.hSMul x (HMul.hMul p₁ p₂))
  -/
  induction p₁, p₂ using induction_on₂ with | _ a₁ s₁ a₂ s₂ => _
  /-
    case h
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    T : Type u_1
    inst✝⁴ : CommSemiring T
    inst✝³ : Algebra R T
    inst✝² : IsLocalization S T
    A : Type u_2
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : T
    a₁ s₁ : A
    a₂ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HMul.hMul (HSMul.hSMul x (LocalizedModule.mk a₁ a₂)) (LocalizedModule.mk …
  -/
  rw [mk_mul_mk, smul_def, smul_def, mk_mul_mk, mul_assoc, smul_mul_assoc]
  /-
    🎉 no goals
  -/


theorem mul_smul' {A : Type*} [Semiring A] [Algebra R A] (x : T) (p₁ p₂ : LocalizedModule S A) :
    p₁ * x • p₂ = x • (p₁ * p₂) := by
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    T : Type u_1
    inst✝⁴ : CommSemiring T
    inst✝³ : Algebra R T
    inst✝² : IsLocalization S T
    A : Type u_2
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : T
    p₁ p₂ : LocalizedModule S A
    ⊢ Eq (HMul.hMul p₁ (HSMul.hSMul x p₂)) (HSMul.hSMul x (HMul.hMul p₁ p₂))
  -/
  induction p₁, p₂ using induction_on₂ with | _ a₁ s₁ a₂ s₂ => _
  /-
    case h
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    T : Type u_1
    inst✝⁴ : CommSemiring T
    inst✝³ : Algebra R T
    inst✝² : IsLocalization S T
    A : Type u_2
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : T
    a₁ s₁ : A
    a₂ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HMul.hMul (LocalizedModule.mk a₁ a₂) (HSMul.hSMul x (LocalizedModule.mk  …
  -/
  rw [smul_def, mk_mul_mk, mk_mul_mk, smul_def, mul_left_comm, mul_smul_comm]
  /-
    🎉 no goals
  -/


noncomputable instance {A : Type*} [Semiring A] [Algebra R A] : Algebra T (LocalizedModule S A) :=
  Algebra.ofModule smul'_mul mul_smul'


theorem algebraMap_mk' {A : Type*} [Semiring A] [Algebra R A] (a : R) (s : S) :
    algebraMap _ _ (IsLocalization.mk' T a s) = mk (algebraMap R A a) s := by
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    T : Type u_1
    inst✝⁴ : CommSemiring T
    inst✝³ : Algebra R T
    inst✝² : IsLocalization S T
    A : Type u_2
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    a : R
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq ((algebraMap T (LocalizedModule S A)) (IsLocalization.mk' T a s)) (Locali …
  -/
  rw [Algebra.algebraMap_eq_smul_one]
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    T : Type u_1
    inst✝⁴ : CommSemiring T
    inst✝³ : Algebra R T
    inst✝² : IsLocalization S T
    A : Type u_2
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    a : R
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (IsLocalization.mk' T a s) 1) (LocalizedModule.mk ((algebraM …
  -/
  change _ • mk _ _ = _
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    T : Type u_1
    inst✝⁴ : CommSemiring T
    inst✝³ : Algebra R T
    inst✝² : IsLocalization S T
    A : Type u_2
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    a : R
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (IsLocalization.mk' T a s) (LocalizedModule.mk 1 1)) (Locali …
  -/
  rw [mk'_smul_mk, Algebra.algebraMap_eq_smul_one, mul_one]
  /-
    🎉 no goals
  -/


theorem algebraMap_mk {A : Type*} [Semiring A] [Algebra R A] (a : R) (s : S) :
    algebraMap _ _ (Localization.mk a s) = mk (algebraMap R A a) s := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    A : Type u_2
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    a : R
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq ((algebraMap (Localization S) (LocalizedModule S A)) (Localization.mk a s …
  -/
  rw [Localization.mk_eq_mk']
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Submonoid R
    A : Type u_2
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    a : R
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq ((algebraMap (Localization S) (LocalizedModule S A)) (IsLocalization.mk'  …
  -/
  exact algebraMap_mk' ..
  /-
    🎉 no goals
  -/


instance : IsScalarTower R T (LocalizedModule S M) where
  smul_assoc r x p := by
    /-
      R : Type u
      inst✝⁵ : CommSemiring R
      S : Submonoid R
      M : Type v
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      T : Type u_1
      inst✝² : CommSemiring T
      inst✝¹ : Algebra R T
      inst✝ : IsLocalization S T
      r : R
      x : T
      p : LocalizedModule S M
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul r x) p) (HSMul.hSMul r (HSMul.hSMul x p))
    -/
    induction' p with m s
    rw [← IsLocalization.mk'_sec (M := S) T x, IsLocalization.smul_mk', mk'_smul_mk, mk'_smul_mk,
      smul'_mk, mul_smul]


noncomputable instance algebra' {A : Type*} [Semiring A] [Algebra R A] :
    Algebra R (LocalizedModule S A) :=
  { (algebraMap (Localization S) (LocalizedModule S A)).comp (algebraMap R <| Localization S),
                                           /-
                                             R : Type u
                                             inst✝⁷ : CommSemiring R
                                             S : Submonoid R
                                             M : Type v
                                             inst✝⁶ : AddCommMonoid M
                                             inst✝⁵ : Module R M
                                             T : Type u_1
                                             inst✝⁴ : CommSemiring T
                                             inst✝³ : Algebra R T
                                             inst✝² : IsLocalization S T
                                             A : Type u_2
                                             inst✝¹ : Semiring A
                                             inst✝ : Algebra R A
                                             ⊢ Module R (LocalizedModule S A)
                                           -/
    show Module R (LocalizedModule S A) by infer_instance with
                                           /-
                                             🎉 no goals
                                           -/
    commutes' := by
      /-
        R : Type u
        inst✝⁷ : CommSemiring R
        S : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        ⊢ ∀ (r : R) (x : LocalizedModule S A), Eq (HMul.hMul (__src✝¹ r) x) (HMul.hMul …
      -/
      intro r x
      /-
        R : Type u
        inst✝⁷ : CommSemiring R
        S : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        r : R
        x : LocalizedModule S A
        ⊢ Eq (HMul.hMul (__src✝¹ r) x) (HMul.hMul x (__src✝¹ r))
      -/
      induction x using induction_on with | _ a s => _
      /-
        case h
        R : Type u
        inst✝⁷ : CommSemiring R
        S : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        r : R
        a : A
        s : Subtype fun x => Membership.mem S x
        ⊢ Eq (HMul.hMul (__src✝¹ r) (LocalizedModule.mk a s)) (HMul.hMul (LocalizedMod …
      -/
      dsimp
      rw [← Localization.mk_one_eq_algebraMap, algebraMap_mk, mk_mul_mk, mk_mul_mk, mul_comm,
        Algebra.commutes]
    smul_def' := by
      /-
        R : Type u
        inst✝⁷ : CommSemiring R
        S : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        ⊢ ∀ (r : R) (x : LocalizedModule S A), Eq (HSMul.hSMul r x) (HMul.hMul (__src✝ …
      -/
      intro r x
      /-
        R : Type u
        inst✝⁷ : CommSemiring R
        S : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        r : R
        x : LocalizedModule S A
        ⊢ Eq (HSMul.hSMul r x) (HMul.hMul (__src✝¹ r) x)
      -/
      induction x using induction_on with | _ a s => _
      /-
        case h
        R : Type u
        inst✝⁷ : CommSemiring R
        S : Submonoid R
        M : Type v
        inst✝⁶ : AddCommMonoid M
        inst✝⁵ : Module R M
        T : Type u_1
        inst✝⁴ : CommSemiring T
        inst✝³ : Algebra R T
        inst✝² : IsLocalization S T
        A : Type u_2
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        r : R
        a : A
        s : Subtype fun x => Membership.mem S x
        ⊢ Eq (HSMul.hSMul r (LocalizedModule.mk a s)) (HMul.hMul (__src✝¹ r) (Localize …
      -/
      dsimp
      rw [← Localization.mk_one_eq_algebraMap, algebraMap_mk, mk_mul_mk, smul'_mk,
        Algebra.smul_def, one_mul] }


/-- The function `m ↦ m / 1` as an `R`-linear map.
-/
@[simps]
def mkLinearMap : M →ₗ[R] LocalizedModule S M where
  toFun m := mk m 1
                     /-
                       R : Type u
                       inst✝⁵ : CommSemiring R
                       S : Submonoid R
                       M : Type v
                       inst✝⁴ : AddCommMonoid M
                       inst✝³ : Module R M
                       T : Type u_1
                       inst✝² : CommSemiring T
                       inst✝¹ : Algebra R T
                       inst✝ : IsLocalization S T
                       x y : M
                       ⊢ Eq ((fun m => LocalizedModule.mk m 1) (HAdd.hAdd x y)) (HAdd.hAdd ((fun m => …
                     -/
  map_add' x y := by simp [mk_add_mk]
                     /-
                       🎉 no goals
                     -/
  map_smul' _ _ := (smul'_mk _ _ _).symm


/-- For any `s : S`, there is an `R`-linear map given by `a/b ↦ a/(b*s)`.
-/
@[simps]
def divBy (s : S) : LocalizedModule S M →ₗ[R] LocalizedModule S M where
  toFun p :=
    p.liftOn (fun p => mk p.1 (p.2 * s)) fun ⟨a, b⟩ ⟨a', b'⟩ ⟨c, eq1⟩ =>
      mk_eq.mpr ⟨c, by rw [mul_smul, mul_smul, smul_comm _ s, smul_comm _ s, eq1, smul_comm _ s,
        smul_comm _ s]⟩
  map_add' x y := by
    /-
      R : Type u
      inst✝⁵ : CommSemiring R
      S : Submonoid R
      M : Type v
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      T : Type u_1
      inst✝² : CommSemiring T
      inst✝¹ : Algebra R T
      inst✝ : IsLocalization S T
      s : Subtype fun x => Membership.mem S x
      x y : LocalizedModule S M
      ⊢ Eq ((fun p => p.liftOn (fun p => LocalizedModule.mk p.1 (HMul.hMul p.2 s)) ⋯ …
    -/
    refine x.induction_on₂ ?_ y
    /-
      R : Type u
      inst✝⁵ : CommSemiring R
      S : Submonoid R
      M : Type v
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      T : Type u_1
      inst✝² : CommSemiring T
      inst✝¹ : Algebra R T
      inst✝ : IsLocalization S T
      s : Subtype fun x => Membership.mem S x
      x y : LocalizedModule S M
      ⊢ ∀ (m m' : M) (s_1 s' : Subtype fun x => Membership.mem S x), Eq ((fun p => p …
    -/
    intro m₁ m₂ t₁ t₂
    simp_rw [mk_add_mk, LocalizedModule.liftOn_mk, mk_add_mk, mul_smul, mul_comm _ s, mul_assoc,
      smul_comm _ s, ← smul_add, mul_left_comm s t₁ t₂, mk_cancel_common_left s]
  map_smul' r x := by
    /-
      R : Type u
      inst✝⁵ : CommSemiring R
      S : Submonoid R
      M : Type v
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      T : Type u_1
      inst✝² : CommSemiring T
      inst✝¹ : Algebra R T
      inst✝ : IsLocalization S T
      s : Subtype fun x => Membership.mem S x
      r : R
      x : LocalizedModule S M
      ⊢ Eq ({ toFun := fun p => p.liftOn (fun p => LocalizedModule.mk p.1 (HMul.hMul …
    -/
    refine x.induction_on (fun _ _ ↦ ?_)
    /-
      R : Type u
      inst✝⁵ : CommSemiring R
      S : Submonoid R
      M : Type v
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      T : Type u_1
      inst✝² : CommSemiring T
      inst✝¹ : Algebra R T
      inst✝ : IsLocalization S T
      s : Subtype fun x => Membership.mem S x
      r : R
      x : LocalizedModule S M
      x✝¹ : M
      x✝ : Subtype fun x => Membership.mem S x
      ⊢ Eq ({ toFun := fun p => p.liftOn (fun p => LocalizedModule.mk p.1 (HMul.hMul …
    -/
    dsimp only
    /-
      R : Type u
      inst✝⁵ : CommSemiring R
      S : Submonoid R
      M : Type v
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      T : Type u_1
      inst✝² : CommSemiring T
      inst✝¹ : Algebra R T
      inst✝ : IsLocalization S T
      s : Subtype fun x => Membership.mem S x
      r : R
      x : LocalizedModule S M
      x✝¹ : M
      x✝ : Subtype fun x => Membership.mem S x
      ⊢ Eq ((HSMul.hSMul r (LocalizedModule.mk x✝¹ x✝)).liftOn (fun p => LocalizedMo …
    -/
    change liftOn (mk _ _) _ _ = r • (liftOn (mk _ _) _ _)
    /-
      R : Type u
      inst✝⁵ : CommSemiring R
      S : Submonoid R
      M : Type v
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      T : Type u_1
      inst✝² : CommSemiring T
      inst✝¹ : Algebra R T
      inst✝ : IsLocalization S T
      s : Subtype fun x => Membership.mem S x
      r : R
      x : LocalizedModule S M
      x✝¹ : M
      x✝ : Subtype fun x => Membership.mem S x
      ⊢ Eq ((LocalizedModule.mk (HSMul.hSMul (IsLocalization.sec S (↑(algebraMap R ( …
    -/
    simp_rw [liftOn_mk, mul_assoc, ← smul_def]
    /-
      R : Type u
      inst✝⁵ : CommSemiring R
      S : Submonoid R
      M : Type v
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      T : Type u_1
      inst✝² : CommSemiring T
      inst✝¹ : Algebra R T
      inst✝ : IsLocalization S T
      s : Subtype fun x => Membership.mem S x
      r : R
      x : LocalizedModule S M
      x✝¹ : M
      x✝ : Subtype fun x => Membership.mem S x
      ⊢ Eq (HSMul.hSMul ({ toFun := (↑↑(algebraMap R (Localization S))).toFun, map_z …
    -/
    congr!
    /-
      🎉 no goals
    -/


theorem divBy_mul_by (s : S) (p : LocalizedModule S M) :
    divBy s (algebraMap R (Module.End R (LocalizedModule S M)) s p) = p :=
  p.induction_on fun m t => by
    /-
      R : Type u
      inst✝² : CommSemiring R
      S : Submonoid R
      M : Type v
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Subtype fun x => Membership.mem S x
      p : LocalizedModule S M
      m : M
      t : Subtype fun x => Membership.mem S x
      ⊢ Eq ((LocalizedModule.divBy s) (((algebraMap R (Module.End R (LocalizedModule …
    -/
    rw [Module.algebraMap_end_apply, divBy_apply]
    /-
      R : Type u
      inst✝² : CommSemiring R
      S : Submonoid R
      M : Type v
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Subtype fun x => Membership.mem S x
      p : LocalizedModule S M
      m : M
      t : Subtype fun x => Membership.mem S x
      ⊢ Eq ((HSMul.hSMul (↑s) (LocalizedModule.mk m t)).liftOn (fun p => LocalizedMo …
    -/
    erw [smul_def]
    /-
      R : Type u
      inst✝² : CommSemiring R
      S : Submonoid R
      M : Type v
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Subtype fun x => Membership.mem S x
      p : LocalizedModule S M
      m : M
      t : Subtype fun x => Membership.mem S x
      ⊢ Eq ((LocalizedModule.mk (HSMul.hSMul (IsLocalization.sec S (↑(algebraMap R ( …
    -/
    rw [LocalizedModule.liftOn_mk, mul_assoc, ← smul_def]
    /-
      R : Type u
      inst✝² : CommSemiring R
      S : Submonoid R
      M : Type v
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Subtype fun x => Membership.mem S x
      p : LocalizedModule S M
      m : M
      t : Subtype fun x => Membership.mem S x
      ⊢ Eq (HSMul.hSMul (↑(algebraMap R (Localization S)).toMonoidWithZeroHom ↑s) (L …
    -/
    erw [smul'_mk]
    /-
      R : Type u
      inst✝² : CommSemiring R
      S : Submonoid R
      M : Type v
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Subtype fun x => Membership.mem S x
      p : LocalizedModule S M
      m : M
      t : Subtype fun x => Membership.mem S x
      ⊢ Eq (LocalizedModule.mk (HSMul.hSMul (↑s) m) (HMul.hMul t s)) (LocalizedModul …
    -/
    rw [← Submonoid.smul_def, mk_cancel_common_right _ s]
    /-
      🎉 no goals
    -/


theorem mul_by_divBy (s : S) (p : LocalizedModule S M) :
    algebraMap R (Module.End R (LocalizedModule S M)) s (divBy s p) = p :=
  p.induction_on fun m t => by
    rw [divBy_apply, Module.algebraMap_end_apply, LocalizedModule.liftOn_mk, smul'_mk,
      ← Submonoid.smul_def, mk_cancel_common_right _ s]


/-- The characteristic predicate for localized module.
`IsLocalizedModule S f` describes that `f : M ⟶ M'` is the localization map identifying `M'` as
`LocalizedModule S M`.
-/
@[mk_iff] class IsLocalizedModule : Prop where
  map_units : ∀ x : S, IsUnit (algebraMap R (Module.End R M') x)
  surj' : ∀ y : M', ∃ x : M × S, x.2 • y = f x.1
  exists_of_eq : ∀ {x₁ x₂}, f x₁ = f x₂ → ∃ c : S, c • x₁ = c • x₂


lemma IsLocalizedModule.surj [IsLocalizedModule S f] (y : M') : ∃ x : M × S, x.2 • y = f x.1 :=
  surj' y

-- Porting note: Manually added to make `S` and `f` explicit.

lemma IsLocalizedModule.eq_iff_exists [IsLocalizedModule S f] {x₁ x₂} :
    f x₁ = f x₂ ↔ ∃ c : S, c • x₁ = c • x₂ :=
  Iff.intro exists_of_eq fun ⟨c, h⟩ ↦ by
    /-
      R : Type u_1
      inst✝⁵ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      inst✝⁴ : AddCommMonoid M
      inst✝³ : AddCommMonoid M'
      inst✝² : Module R M
      inst✝¹ : Module R M'
      f : LinearMap (RingHom.id R) M M'
      inst✝ : IsLocalizedModule S f
      x₁ x₂ : M
      x✝ : Exists fun c => Eq (HSMul.hSMul c x₁) (HSMul.hSMul c x₂)
      c : Subtype fun x => Membership.mem S x
      h : Eq (HSMul.hSMul c x₁) (HSMul.hSMul c x₂)
      ⊢ Eq (f x₁) (f x₂)
    -/
    apply_fun f at h
    /-
      R : Type u_1
      inst✝⁵ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      inst✝⁴ : AddCommMonoid M
      inst✝³ : AddCommMonoid M'
      inst✝² : Module R M
      inst✝¹ : Module R M'
      f : LinearMap (RingHom.id R) M M'
      inst✝ : IsLocalizedModule S f
      x₁ x₂ : M
      x✝ : Exists fun c => Eq (HSMul.hSMul c x₁) (HSMul.hSMul c x₂)
      c : Subtype fun x => Membership.mem S x
      h : Eq (f (HSMul.hSMul c x₁)) (f (HSMul.hSMul c x₂))
      ⊢ Eq (f x₁) (f x₂)
    -/
    simp_rw [f.map_smul_of_tower, Submonoid.smul_def, ← Module.algebraMap_end_apply R R] at h
    /-
      R : Type u_1
      inst✝⁵ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      inst✝⁴ : AddCommMonoid M
      inst✝³ : AddCommMonoid M'
      inst✝² : Module R M
      inst✝¹ : Module R M'
      f : LinearMap (RingHom.id R) M M'
      inst✝ : IsLocalizedModule S f
      x₁ x₂ : M
      x✝ : Exists fun c => Eq (HSMul.hSMul c x₁) (HSMul.hSMul c x₂)
      c : Subtype fun x => Membership.mem S x
      h : Eq (((algebraMap R (Module.End R M')) ↑c) (f x₁)) (((algebraMap R (Module. …
      ⊢ Eq (f x₁) (f x₂)
    -/
    exact ((Module.End_isUnit_iff _).mp <| map_units f c).1 h
    /-
      🎉 no goals
    -/


instance IsLocalizedModule.of_linearEquiv (e : M' ≃ₗ[R] M'') [hf : IsLocalizedModule S f] :
    IsLocalizedModule S (e ∘ₗ f : M →ₗ[R] M'') where
  map_units s := by
    rw [show algebraMap R (Module.End R M'') s = e ∘ₗ (algebraMap R (Module.End R M') s) ∘ₗ e.symm
      by ext; simp, Module.End_isUnit_iff, LinearMap.coe_comp, LinearMap.coe_comp,
      LinearEquiv.coe_coe, LinearEquiv.coe_coe, EquivLike.comp_bijective, EquivLike.bijective_comp]
    /-
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : AddCommMonoid M'
      inst✝⁸ : AddCommMonoid M''
      A : Type u_5
      inst✝⁷ : CommSemiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : Module A M'
      inst✝⁴ : IsLocalization S A
      inst✝³ : Module R M
      inst✝² : Module R M'
      inst✝¹ : Module R M''
      inst✝ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g : LinearMap (RingHom.id R) M M''
      e : LinearEquiv (RingHom.id R) M' M''
      hf : IsLocalizedModule S f
      s : Subtype fun x => Membership.mem S x
      ⊢ Function.Bijective ⇑((algebraMap R (Module.End R M')) ↑s)
    -/
    exact (Module.End_isUnit_iff _).mp <| hf.map_units s
    /-
      🎉 no goals
    -/
  surj' x := by
    /-
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : AddCommMonoid M'
      inst✝⁸ : AddCommMonoid M''
      A : Type u_5
      inst✝⁷ : CommSemiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : Module A M'
      inst✝⁴ : IsLocalization S A
      inst✝³ : Module R M
      inst✝² : Module R M'
      inst✝¹ : Module R M''
      inst✝ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g : LinearMap (RingHom.id R) M M''
      e : LinearEquiv (RingHom.id R) M' M''
      hf : IsLocalizedModule S f
      x : M''
      ⊢ Exists fun x_1 => Eq (HSMul.hSMul x_1.2 x) (((↑e).comp f) x_1.1)
    -/
    obtain ⟨p, h⟩ := hf.surj' (e.symm x)
    exact ⟨p, by rw [LinearMap.coe_comp, LinearEquiv.coe_coe, Function.comp_apply, ← e.congr_arg h,
      Submonoid.smul_def, Submonoid.smul_def, LinearEquiv.map_smul, LinearEquiv.apply_symm_apply]⟩
  exists_of_eq h := by
    simp_rw [LinearMap.coe_comp, LinearEquiv.coe_coe, Function.comp_apply,
      EmbeddingLike.apply_eq_iff_eq] at h
    /-
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : AddCommMonoid M'
      inst✝⁸ : AddCommMonoid M''
      A : Type u_5
      inst✝⁷ : CommSemiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : Module A M'
      inst✝⁴ : IsLocalization S A
      inst✝³ : Module R M
      inst✝² : Module R M'
      inst✝¹ : Module R M''
      inst✝ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g : LinearMap (RingHom.id R) M M''
      e : LinearEquiv (RingHom.id R) M' M''
      hf : IsLocalizedModule S f
      x₁✝ x₂✝ : M
      h : Eq (f x₁✝) (f x₂✝)
      ⊢ Exists fun c => Eq (HSMul.hSMul c x₁✝) (HSMul.hSMul c x₂✝)
    -/
    exact hf.exists_of_eq h
    /-
      🎉 no goals
    -/


instance IsLocalizedModule.of_linearEquiv_right (e : M'' ≃ₗ[R] M) [hf : IsLocalizedModule S f] :
    IsLocalizedModule S (f ∘ₗ e : M'' →ₗ[R] M') where
  map_units s := hf.map_units s
  surj' x := by
    /-
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : AddCommMonoid M'
      inst✝⁸ : AddCommMonoid M''
      A : Type u_5
      inst✝⁷ : CommSemiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : Module A M'
      inst✝⁴ : IsLocalization S A
      inst✝³ : Module R M
      inst✝² : Module R M'
      inst✝¹ : Module R M''
      inst✝ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g : LinearMap (RingHom.id R) M M''
      e : LinearEquiv (RingHom.id R) M'' M
      hf : IsLocalizedModule S f
      x : M'
      ⊢ Exists fun x_1 => Eq (HSMul.hSMul x_1.2 x) ((f.comp ↑e) x_1.1)
    -/
    obtain ⟨⟨p, s⟩, h⟩ := hf.surj' x
    /-
      case intro.mk
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : AddCommMonoid M'
      inst✝⁸ : AddCommMonoid M''
      A : Type u_5
      inst✝⁷ : CommSemiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : Module A M'
      inst✝⁴ : IsLocalization S A
      inst✝³ : Module R M
      inst✝² : Module R M'
      inst✝¹ : Module R M''
      inst✝ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g : LinearMap (RingHom.id R) M M''
      e : LinearEquiv (RingHom.id R) M'' M
      hf : IsLocalizedModule S f
      x : M'
      p : M
      s : Subtype fun x => Membership.mem S x
      h : Eq (HSMul.hSMul { fst := p, snd := s }.2 x) (f { fst := p, snd := s }.1)
      ⊢ Exists fun x_1 => Eq (HSMul.hSMul x_1.2 x) ((f.comp ↑e) x_1.1)
    -/
    exact ⟨⟨e.symm p, s⟩, by simpa using h⟩
    /-
      🎉 no goals
    -/
  exists_of_eq h := by
    /-
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : AddCommMonoid M'
      inst✝⁸ : AddCommMonoid M''
      A : Type u_5
      inst✝⁷ : CommSemiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : Module A M'
      inst✝⁴ : IsLocalization S A
      inst✝³ : Module R M
      inst✝² : Module R M'
      inst✝¹ : Module R M''
      inst✝ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g : LinearMap (RingHom.id R) M M''
      e : LinearEquiv (RingHom.id R) M'' M
      hf : IsLocalizedModule S f
      x₁✝ x₂✝ : M''
      h : Eq ((f.comp ↑e) x₁✝) ((f.comp ↑e) x₂✝)
      ⊢ Exists fun c => Eq (HSMul.hSMul c x₁✝) (HSMul.hSMul c x₂✝)
    -/
    simp_rw [LinearMap.coe_comp, LinearEquiv.coe_coe, Function.comp_apply] at h
    /-
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : AddCommMonoid M'
      inst✝⁸ : AddCommMonoid M''
      A : Type u_5
      inst✝⁷ : CommSemiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : Module A M'
      inst✝⁴ : IsLocalization S A
      inst✝³ : Module R M
      inst✝² : Module R M'
      inst✝¹ : Module R M''
      inst✝ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g : LinearMap (RingHom.id R) M M''
      e : LinearEquiv (RingHom.id R) M'' M
      hf : IsLocalizedModule S f
      x₁✝ x₂✝ : M''
      h : Eq (f (e x₁✝)) (f (e x₂✝))
      ⊢ Exists fun c => Eq (HSMul.hSMul c x₁✝) (HSMul.hSMul c x₂✝)
    -/
    obtain ⟨c, hc⟩ := hf.exists_of_eq h
    exact ⟨c, by simpa only [Submonoid.smul_def, map_smul, e.symm_apply_apply]
      using congr(e.symm $hc)⟩


variable (M) in
lemma isLocalizedModule_id (R') [CommSemiring R'] [Algebra R R'] [IsLocalization S R'] [Module R' M]
    [IsScalarTower R R' M] : IsLocalizedModule S (.id : M →ₗ[R] M) where
  map_units s := by
    /-
      R : Type u_1
      inst✝⁷ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      R' : Type u_6
      inst✝⁴ : CommSemiring R'
      inst✝³ : Algebra R R'
      inst✝² : IsLocalization S R'
      inst✝¹ : Module R' M
      inst✝ : IsScalarTower R R' M
      s : Subtype fun x => Membership.mem S x
      ⊢ IsUnit ((algebraMap R (Module.End R M)) ↑s)
    -/
    rw [← (Algebra.lsmul R (A := R') R M).commutes]; exact (IsLocalization.map_units R' s).map _
                                                     /-
                                                       🎉 no goals
                                                     -/
  surj' m := ⟨(m, 1), one_smul _ _⟩
  exists_of_eq h := ⟨1, congr_arg _ h⟩


/--
If `g` is a linear map `M → M''` such that all scalar multiplication by `s : S` is invertible, then
there is a linear map `LocalizedModule S M → M''`.
-/
noncomputable def lift' (g : M →ₗ[R] M'')
    (h : ∀ x : S, IsUnit (algebraMap R (Module.End R M'') x)) : LocalizedModule S M → M'' :=
  fun m =>
  m.liftOn (fun p => (h p.2).unit⁻¹.val <| g p.1) fun ⟨m, s⟩ ⟨m', s'⟩ ⟨c, eq1⟩ => by
    -- Porting note: We remove `generalize_proofs h1 h2`. This does nothing here.
    /-
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : AddCommMonoid M'
      inst✝⁸ : AddCommMonoid M''
      A : Type u_5
      inst✝⁷ : CommSemiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : Module A M'
      inst✝⁴ : IsLocalization S A
      inst✝³ : Module R M
      inst✝² : Module R M'
      inst✝¹ : Module R M''
      inst✝ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g✝ g : LinearMap (RingHom.id R) M M''
      h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
      m✝ : LocalizedModule S M
      x✝² x✝¹ : Prod M (Subtype fun x => Membership.mem S x)
      m : M
      s : Subtype fun x => Membership.mem S x
      m' : M
      s' : Subtype fun x => Membership.mem S x
      x✝ : HasEquiv.Equiv { fst := m, snd := s } { fst := m', snd := s' }
      c : Subtype fun x => Membership.mem S x
      eq1 : Eq (HSMul.hSMul c (HSMul.hSMul { fst := m', snd := s' }.2 { fst := m, sn …
      ⊢ Eq ((fun p => ↑(Inv.inv ⋯.unit) (g p.1)) { fst := m, snd := s }) ((fun p =>  …
    -/
    dsimp only
    /-
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : AddCommMonoid M'
      inst✝⁸ : AddCommMonoid M''
      A : Type u_5
      inst✝⁷ : CommSemiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : Module A M'
      inst✝⁴ : IsLocalization S A
      inst✝³ : Module R M
      inst✝² : Module R M'
      inst✝¹ : Module R M''
      inst✝ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g✝ g : LinearMap (RingHom.id R) M M''
      h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
      m✝ : LocalizedModule S M
      x✝² x✝¹ : Prod M (Subtype fun x => Membership.mem S x)
      m : M
      s : Subtype fun x => Membership.mem S x
      m' : M
      s' : Subtype fun x => Membership.mem S x
      x✝ : HasEquiv.Equiv { fst := m, snd := s } { fst := m', snd := s' }
      c : Subtype fun x => Membership.mem S x
      eq1 : Eq (HSMul.hSMul c (HSMul.hSMul { fst := m', snd := s' }.2 { fst := m, sn …
      ⊢ Eq (↑(Inv.inv ⋯.unit) (g m)) (↑(Inv.inv ⋯.unit) (g m'))
    -/
    simp only [Submonoid.smul_def] at eq1
    rw [Module.End_algebraMap_isUnit_inv_apply_eq_iff, ← map_smul, eq_comm,
      Module.End_algebraMap_isUnit_inv_apply_eq_iff]
    have : c • s • g m' = c • s' • g m := by
      simp only [Submonoid.smul_def, ← g.map_smul, eq1]
    /-
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : AddCommMonoid M'
      inst✝⁸ : AddCommMonoid M''
      A : Type u_5
      inst✝⁷ : CommSemiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : Module A M'
      inst✝⁴ : IsLocalization S A
      inst✝³ : Module R M
      inst✝² : Module R M'
      inst✝¹ : Module R M''
      inst✝ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g✝ g : LinearMap (RingHom.id R) M M''
      h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
      m✝ : LocalizedModule S M
      x✝² x✝¹ : Prod M (Subtype fun x => Membership.mem S x)
      m : M
      s : Subtype fun x => Membership.mem S x
      m' : M
      s' : Subtype fun x => Membership.mem S x
      x✝ : HasEquiv.Equiv { fst := m, snd := s } { fst := m', snd := s' }
      c : Subtype fun x => Membership.mem S x
      eq1 : Eq (HSMul.hSMul (↑c) (HSMul.hSMul (↑s') m)) (HSMul.hSMul (↑c) (HSMul.hSM …
      this : Eq (HSMul.hSMul c (HSMul.hSMul s (g m'))) (HSMul.hSMul c (HSMul.hSMul s …
      ⊢ Eq (HSMul.hSMul (↑s) (g m')) (HSMul.hSMul (↑s') (g m))
    -/
    have : Function.Injective (h c).unit.inv := ((Module.End_isUnit_iff _).1 (by simp)).1
    /-
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : AddCommMonoid M'
      inst✝⁸ : AddCommMonoid M''
      A : Type u_5
      inst✝⁷ : CommSemiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : Module A M'
      inst✝⁴ : IsLocalization S A
      inst✝³ : Module R M
      inst✝² : Module R M'
      inst✝¹ : Module R M''
      inst✝ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g✝ g : LinearMap (RingHom.id R) M M''
      h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
      m✝ : LocalizedModule S M
      x✝² x✝¹ : Prod M (Subtype fun x => Membership.mem S x)
      m : M
      s : Subtype fun x => Membership.mem S x
      m' : M
      s' : Subtype fun x => Membership.mem S x
      x✝ : HasEquiv.Equiv { fst := m, snd := s } { fst := m', snd := s' }
      c : Subtype fun x => Membership.mem S x
      eq1 : Eq (HSMul.hSMul (↑c) (HSMul.hSMul (↑s') m)) (HSMul.hSMul (↑c) (HSMul.hSM …
      this✝ : Eq (HSMul.hSMul c (HSMul.hSMul s (g m'))) (HSMul.hSMul c (HSMul.hSMul  …
      this : Function.Injective ⇑⋯.unit.inv
      ⊢ Eq (HSMul.hSMul (↑s) (g m')) (HSMul.hSMul (↑s') (g m))
    -/
    apply_fun (h c).unit.inv
    rw [Units.inv_eq_val_inv, Module.End_algebraMap_isUnit_inv_apply_eq_iff, ←
      (h c).unit⁻¹.val.map_smul]
    /-
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : AddCommMonoid M'
      inst✝⁸ : AddCommMonoid M''
      A : Type u_5
      inst✝⁷ : CommSemiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : Module A M'
      inst✝⁴ : IsLocalization S A
      inst✝³ : Module R M
      inst✝² : Module R M'
      inst✝¹ : Module R M''
      inst✝ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g✝ g : LinearMap (RingHom.id R) M M''
      h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
      m✝ : LocalizedModule S M
      x✝² x✝¹ : Prod M (Subtype fun x => Membership.mem S x)
      m : M
      s : Subtype fun x => Membership.mem S x
      m' : M
      s' : Subtype fun x => Membership.mem S x
      x✝ : HasEquiv.Equiv { fst := m, snd := s } { fst := m', snd := s' }
      c : Subtype fun x => Membership.mem S x
      eq1 : Eq (HSMul.hSMul (↑c) (HSMul.hSMul (↑s') m)) (HSMul.hSMul (↑c) (HSMul.hSM …
      this✝ : Eq (HSMul.hSMul c (HSMul.hSMul s (g m'))) (HSMul.hSMul c (HSMul.hSMul  …
      this : Function.Injective ⇑⋯.unit.inv
      ⊢ Eq (HSMul.hSMul (↑s) (g m')) (↑(Inv.inv ⋯.unit) (HSMul.hSMul (↑c) (HSMul.hSM …
    -/
    symm
    rw [Module.End_algebraMap_isUnit_inv_apply_eq_iff, ← g.map_smul, ← g.map_smul, ← g.map_smul, ←
      g.map_smul, eq1]


theorem lift'_mk (g : M →ₗ[R] M'') (h : ∀ x : S, IsUnit ((algebraMap R (Module.End R M'')) x))
    (m : M) (s : S) :
    LocalizedModule.lift' S g h (LocalizedModule.mk m s) = (h s).unit⁻¹.val (g m) :=
  rfl


theorem lift'_add (g : M →ₗ[R] M'') (h : ∀ x : S, IsUnit ((algebraMap R (Module.End R M'')) x))
    (x y) :
    LocalizedModule.lift' S g h (x + y) =
      LocalizedModule.lift' S g h x + LocalizedModule.lift' S g h y :=
  LocalizedModule.induction_on₂
    (by
      /-
        R : Type u_1
        inst✝⁴ : CommSemiring R
        S : Submonoid R
        M : Type u_2
        M'' : Type u_4
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid M''
        inst✝¹ : Module R M
        inst✝ : Module R M''
        g : LinearMap (RingHom.id R) M M''
        h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
        x y : LocalizedModule S M
        ⊢ ∀ (m m' : M) (s s' : Subtype fun x => Membership.mem S x), Eq (LocalizedModu …
      -/
      intro a a' b b'
      /-
        R : Type u_1
        inst✝⁴ : CommSemiring R
        S : Submonoid R
        M : Type u_2
        M'' : Type u_4
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid M''
        inst✝¹ : Module R M
        inst✝ : Module R M''
        g : LinearMap (RingHom.id R) M M''
        h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
        x y : LocalizedModule S M
        a a' : M
        b b' : Subtype fun x => Membership.mem S x
        ⊢ Eq (LocalizedModule.lift' S g h (HAdd.hAdd (LocalizedModule.mk a b) (Localiz …
      -/
      erw [LocalizedModule.lift'_mk, LocalizedModule.lift'_mk, LocalizedModule.lift'_mk]
      -- Porting note: We remove `generalize_proofs h1 h2 h3`. This only generalize `h1`.
      rw [map_add, Module.End_algebraMap_isUnit_inv_apply_eq_iff, smul_add, ← map_smul,
        ← map_smul, ← map_smul]
      /-
        R : Type u_1
        inst✝⁴ : CommSemiring R
        S : Submonoid R
        M : Type u_2
        M'' : Type u_4
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid M''
        inst✝¹ : Module R M
        inst✝ : Module R M''
        g : LinearMap (RingHom.id R) M M''
        h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
        x y : LocalizedModule S M
        a a' : M
        b b' : Subtype fun x => Membership.mem S x
        ⊢ Eq (HAdd.hAdd (g (HSMul.hSMul { fst := a', snd := b' }.2 { fst := a, snd :=  …
      -/
      congr 1 <;> symm
        /-
          case e_a
          R : Type u_1
          inst✝⁴ : CommSemiring R
          S : Submonoid R
          M : Type u_2
          M'' : Type u_4
          inst✝³ : AddCommMonoid M
          inst✝² : AddCommMonoid M''
          inst✝¹ : Module R M
          inst✝ : Module R M''
          g : LinearMap (RingHom.id R) M M''
          h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
          x y : LocalizedModule S M
          a a' : M
          b b' : Subtype fun x => Membership.mem S x
          ⊢ Eq (↑(Inv.inv ⋯.unit) (g (HSMul.hSMul (↑(HMul.hMul { fst := a, snd := b }.2  …
        -/
      · erw [Module.End_algebraMap_isUnit_inv_apply_eq_iff, mul_smul, ← map_smul]
        /-
          case e_a
          R : Type u_1
          inst✝⁴ : CommSemiring R
          S : Submonoid R
          M : Type u_2
          M'' : Type u_4
          inst✝³ : AddCommMonoid M
          inst✝² : AddCommMonoid M''
          inst✝¹ : Module R M
          inst✝ : Module R M''
          g : LinearMap (RingHom.id R) M M''
          h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
          x y : LocalizedModule S M
          a a' : M
          b b' : Subtype fun x => Membership.mem S x
          ⊢ Eq (g (HSMul.hSMul (↑{ fst := a, snd := b }.2) (HSMul.hSMul (↑{ fst := a', s …
        -/
        rfl
        /-
          🎉 no goals
        -/
        /-
          case e_a
          R : Type u_1
          inst✝⁴ : CommSemiring R
          S : Submonoid R
          M : Type u_2
          M'' : Type u_4
          inst✝³ : AddCommMonoid M
          inst✝² : AddCommMonoid M''
          inst✝¹ : Module R M
          inst✝ : Module R M''
          g : LinearMap (RingHom.id R) M M''
          h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
          x y : LocalizedModule S M
          a a' : M
          b b' : Subtype fun x => Membership.mem S x
          ⊢ Eq (↑(Inv.inv ⋯.unit) (HSMul.hSMul (↑(HMul.hMul { fst := a, snd := b }.2 { f …
        -/
      · dsimp
        /-
          case e_a
          R : Type u_1
          inst✝⁴ : CommSemiring R
          S : Submonoid R
          M : Type u_2
          M'' : Type u_4
          inst✝³ : AddCommMonoid M
          inst✝² : AddCommMonoid M''
          inst✝¹ : Module R M
          inst✝ : Module R M''
          g : LinearMap (RingHom.id R) M M''
          h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
          x y : LocalizedModule S M
          a a' : M
          b b' : Subtype fun x => Membership.mem S x
          ⊢ Eq (↑(Inv.inv ⋯.unit) (HSMul.hSMul (HMul.hMul ↑b ↑b') (g a'))) (g (HSMul.hSM …
        -/
        rw [Module.End_algebraMap_isUnit_inv_apply_eq_iff, mul_comm, mul_smul, ← map_smul]
        /-
          case e_a
          R : Type u_1
          inst✝⁴ : CommSemiring R
          S : Submonoid R
          M : Type u_2
          M'' : Type u_4
          inst✝³ : AddCommMonoid M
          inst✝² : AddCommMonoid M''
          inst✝¹ : Module R M
          inst✝ : Module R M''
          g : LinearMap (RingHom.id R) M M''
          h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
          x y : LocalizedModule S M
          a a' : M
          b b' : Subtype fun x => Membership.mem S x
          ⊢ Eq (HSMul.hSMul (↑b') (g (HSMul.hSMul (↑b) a'))) (HSMul.hSMul (↑b') (g (HSMu …
        -/
        rfl)
        /-
          🎉 no goals
        -/
    x y


theorem lift'_smul (g : M →ₗ[R] M'') (h : ∀ x : S, IsUnit ((algebraMap R (Module.End R M'')) x))
    (r : R) (m) : r • LocalizedModule.lift' S g h m = LocalizedModule.lift' S g h (r • m) :=
  m.induction_on fun a b => by
    /-
      R : Type u_1
      inst✝⁴ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M'' : Type u_4
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M''
      inst✝¹ : Module R M
      inst✝ : Module R M''
      g : LinearMap (RingHom.id R) M M''
      h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
      r : R
      m : LocalizedModule S M
      a : M
      b : Subtype fun x => Membership.mem S x
      ⊢ Eq (HSMul.hSMul r (LocalizedModule.lift' S g h (LocalizedModule.mk a b))) (L …
    -/
    rw [LocalizedModule.lift'_mk, LocalizedModule.smul'_mk, LocalizedModule.lift'_mk]
    -- Porting note: We remove `generalize_proofs h1 h2`. This does nothing here.
    /-
      R : Type u_1
      inst✝⁴ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M'' : Type u_4
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M''
      inst✝¹ : Module R M
      inst✝ : Module R M''
      g : LinearMap (RingHom.id R) M M''
      h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
      r : R
      m : LocalizedModule S M
      a : M
      b : Subtype fun x => Membership.mem S x
      ⊢ Eq (HSMul.hSMul r (↑(Inv.inv ⋯.unit) (g a))) (↑(Inv.inv ⋯.unit) (g (HSMul.hS …
    -/
    rw [← map_smul, ← g.map_smul]
    /-
      🎉 no goals
    -/


/--
If `g` is a linear map `M → M''` such that all scalar multiplication by `s : S` is invertible, then
there is a linear map `LocalizedModule S M → M''`.
-/
noncomputable def lift (g : M →ₗ[R] M'')
    (h : ∀ x : S, IsUnit ((algebraMap R (Module.End R M'')) x)) :
    LocalizedModule S M →ₗ[R] M'' where
  toFun := LocalizedModule.lift' S g h
  map_add' := LocalizedModule.lift'_add S g h
                      /-
                        R : Type u_1
                        inst✝¹¹ : CommSemiring R
                        S : Submonoid R
                        M : Type u_2
                        M' : Type u_3
                        M'' : Type u_4
                        inst✝¹⁰ : AddCommMonoid M
                        inst✝⁹ : AddCommMonoid M'
                        inst✝⁸ : AddCommMonoid M''
                        A : Type u_5
                        inst✝⁷ : CommSemiring A
                        inst✝⁶ : Algebra R A
                        inst✝⁵ : Module A M'
                        inst✝⁴ : IsLocalization S A
                        inst✝³ : Module R M
                        inst✝² : Module R M'
                        inst✝¹ : Module R M''
                        inst✝ : IsScalarTower R A M'
                        f : LinearMap (RingHom.id R) M M'
                        g✝ g : LinearMap (RingHom.id R) M M''
                        h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
                        r : R
                        x : LocalizedModule S M
                        ⊢ Eq ({ toFun := LocalizedModule.lift' S g h, map_add' := ⋯ }.toFun (HSMul.hSM …
                      -/
  map_smul' r x := by rw [LocalizedModule.lift'_smul, RingHom.id_apply]
                      /-
                        🎉 no goals
                      -/


/--
If `g` is a linear map `M → M''` such that all scalar multiplication by `s : S` is invertible, then
`lift g m s = s⁻¹ • g m`.
-/
theorem lift_mk
    (g : M →ₗ[R] M'') (h : ∀ x : S, IsUnit (algebraMap R (Module.End R M'') x)) (m : M) (s : S) :
    LocalizedModule.lift S g h (LocalizedModule.mk m s) = (h s).unit⁻¹.val (g m) :=
  rfl


/--
If `g` is a linear map `M → M''` such that all scalar multiplication by `s : S` is invertible, then
there is a linear map `lift g ∘ mkLinearMap = g`.
-/
theorem lift_comp (g : M →ₗ[R] M'') (h : ∀ x : S, IsUnit ((algebraMap R (Module.End R M'')) x)) :
    (lift S g h).comp (mkLinearMap S M) = g := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M'' : Type u_4
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M''
    inst✝¹ : Module R M
    inst✝ : Module R M''
    g : LinearMap (RingHom.id R) M M''
    h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
    ⊢ Eq ((LocalizedModule.lift S g h).comp (LocalizedModule.mkLinearMap S M)) g
  -/
  ext x; dsimp; rw [LocalizedModule.lift_mk]
  /-
    case h
    R : Type u_1
    inst✝⁴ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M'' : Type u_4
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M''
    inst✝¹ : Module R M
    inst✝ : Module R M''
    g : LinearMap (RingHom.id R) M M''
    h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
    x : M
    ⊢ Eq (↑(Inv.inv ⋯.unit) (g x)) (g x)
  -/
  erw [Module.End_algebraMap_isUnit_inv_apply_eq_iff, one_smul]
  /-
    🎉 no goals
  -/


/--
If `g` is a linear map `M → M''` such that all scalar multiplication by `s : S` is invertible and
`l` is another linear map `LocalizedModule S M ⟶ M''` such that `l ∘ mkLinearMap = g` then
`l = lift g`
-/
theorem lift_unique (g : M →ₗ[R] M'') (h : ∀ x : S, IsUnit ((algebraMap R (Module.End R M'')) x))
    (l : LocalizedModule S M →ₗ[R] M'') (hl : l.comp (LocalizedModule.mkLinearMap S M) = g) :
    LocalizedModule.lift S g h = l := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M'' : Type u_4
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M''
    inst✝¹ : Module R M
    inst✝ : Module R M''
    g : LinearMap (RingHom.id R) M M''
    h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
    l : LinearMap (RingHom.id R) (LocalizedModule S M) M''
    hl : Eq (l.comp (LocalizedModule.mkLinearMap S M)) g
    ⊢ Eq (LocalizedModule.lift S g h) l
  -/
  ext x; induction' x with m s
  /-
    case h.h
    R : Type u_1
    inst✝⁴ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M'' : Type u_4
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M''
    inst✝¹ : Module R M
    inst✝ : Module R M''
    g : LinearMap (RingHom.id R) M M''
    h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
    l : LinearMap (RingHom.id R) (LocalizedModule S M) M''
    hl : Eq (l.comp (LocalizedModule.mkLinearMap S M)) g
    m : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq ((LocalizedModule.lift S g h) (LocalizedModule.mk m s)) (l (LocalizedModu …
  -/
  rw [LocalizedModule.lift_mk]
  rw [Module.End_algebraMap_isUnit_inv_apply_eq_iff, ← hl, LinearMap.coe_comp,
    Function.comp_apply, LocalizedModule.mkLinearMap_apply, ← l.map_smul, LocalizedModule.smul'_mk]
  /-
    case h.h
    R : Type u_1
    inst✝⁴ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M'' : Type u_4
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M''
    inst✝¹ : Module R M
    inst✝ : Module R M''
    g : LinearMap (RingHom.id R) M M''
    h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
    l : LinearMap (RingHom.id R) (LocalizedModule S M) M''
    hl : Eq (l.comp (LocalizedModule.mkLinearMap S M)) g
    m : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (l (LocalizedModule.mk m 1)) (l (LocalizedModule.mk (HSMul.hSMul (↑s) m)  …
  -/
  congr 1; rw [LocalizedModule.mk_eq]
  /-
    case h.h.h.e_6.h
    R : Type u_1
    inst✝⁴ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M'' : Type u_4
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M''
    inst✝¹ : Module R M
    inst✝ : Module R M''
    g : LinearMap (RingHom.id R) M M''
    h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
    l : LinearMap (RingHom.id R) (LocalizedModule S M) M''
    hl : Eq (l.comp (LocalizedModule.mkLinearMap S M)) g
    m : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Exists fun u => Eq (HSMul.hSMul u (HSMul.hSMul s m)) (HSMul.hSMul u (HSMul.h …
  -/
  refine ⟨1, ?_⟩; simp only [one_smul, Submonoid.smul_def]
                  /-
                    🎉 no goals
                  -/


instance localizedModuleIsLocalizedModule :
    IsLocalizedModule S (LocalizedModule.mkLinearMap S M) where
  map_units s :=
    ⟨⟨algebraMap R (Module.End R (LocalizedModule S M)) s, LocalizedModule.divBy s,
        DFunLike.ext _ _ <| LocalizedModule.mul_by_divBy s,
        DFunLike.ext _ _ <| LocalizedModule.divBy_mul_by s⟩,
      DFunLike.ext _ _ fun p =>
        p.induction_on <| by
          /-
            R : Type u_1
            inst✝¹¹ : CommSemiring R
            S : Submonoid R
            M : Type u_2
            M' : Type u_3
            M'' : Type u_4
            inst✝¹⁰ : AddCommMonoid M
            inst✝⁹ : AddCommMonoid M'
            inst✝⁸ : AddCommMonoid M''
            A : Type u_5
            inst✝⁷ : CommSemiring A
            inst✝⁶ : Algebra R A
            inst✝⁵ : Module A M'
            inst✝⁴ : IsLocalization S A
            inst✝³ : Module R M
            inst✝² : Module R M'
            inst✝¹ : Module R M''
            inst✝ : IsScalarTower R A M'
            f : LinearMap (RingHom.id R) M M'
            g : LinearMap (RingHom.id R) M M''
            s : Subtype fun x => Membership.mem S x
            p : LocalizedModule S M
            ⊢ ∀ (m : M) (s_1 : Subtype fun x => Membership.mem S x), Eq (↑{ val := (algebr …
          -/
          intros
          /-
            R : Type u_1
            inst✝¹¹ : CommSemiring R
            S : Submonoid R
            M : Type u_2
            M' : Type u_3
            M'' : Type u_4
            inst✝¹⁰ : AddCommMonoid M
            inst✝⁹ : AddCommMonoid M'
            inst✝⁸ : AddCommMonoid M''
            A : Type u_5
            inst✝⁷ : CommSemiring A
            inst✝⁶ : Algebra R A
            inst✝⁵ : Module A M'
            inst✝⁴ : IsLocalization S A
            inst✝³ : Module R M
            inst✝² : Module R M'
            inst✝¹ : Module R M''
            inst✝ : IsScalarTower R A M'
            f : LinearMap (RingHom.id R) M M'
            g : LinearMap (RingHom.id R) M M''
            s : Subtype fun x => Membership.mem S x
            p : LocalizedModule S M
            m✝ : M
            s✝ : Subtype fun x => Membership.mem S x
            ⊢ Eq (↑{ val := (algebraMap R (Module.End R (LocalizedModule S M))) ↑s, inv := …
          -/
          rfl⟩
          /-
            🎉 no goals
          -/
  surj' p :=
    p.induction_on fun m t => by
      /-
        R : Type u_1
        inst✝¹¹ : CommSemiring R
        S : Submonoid R
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : AddCommMonoid M'
        inst✝⁸ : AddCommMonoid M''
        A : Type u_5
        inst✝⁷ : CommSemiring A
        inst✝⁶ : Algebra R A
        inst✝⁵ : Module A M'
        inst✝⁴ : IsLocalization S A
        inst✝³ : Module R M
        inst✝² : Module R M'
        inst✝¹ : Module R M''
        inst✝ : IsScalarTower R A M'
        f : LinearMap (RingHom.id R) M M'
        g : LinearMap (RingHom.id R) M M''
        p : LocalizedModule S M
        m : M
        t : Subtype fun x => Membership.mem S x
        ⊢ Exists fun x => Eq (HSMul.hSMul x.2 (LocalizedModule.mk m t)) ((LocalizedMod …
      -/
      refine ⟨⟨m, t⟩, ?_⟩
      erw [LocalizedModule.smul'_mk, LocalizedModule.mkLinearMap_apply, Submonoid.coe_subtype,
        LocalizedModule.mk_cancel t]
                         /-
                           R : Type u_1
                           inst✝¹¹ : CommSemiring R
                           S : Submonoid R
                           M : Type u_2
                           M' : Type u_3
                           M'' : Type u_4
                           inst✝¹⁰ : AddCommMonoid M
                           inst✝⁹ : AddCommMonoid M'
                           inst✝⁸ : AddCommMonoid M''
                           A : Type u_5
                           inst✝⁷ : CommSemiring A
                           inst✝⁶ : Algebra R A
                           inst✝⁵ : Module A M'
                           inst✝⁴ : IsLocalization S A
                           inst✝³ : Module R M
                           inst✝² : Module R M'
                           inst✝¹ : Module R M''
                           inst✝ : IsScalarTower R A M'
                           f : LinearMap (RingHom.id R) M M'
                           g : LinearMap (RingHom.id R) M M''
                           x₁✝ x₂✝ : M
                           eq1 : Eq ((LocalizedModule.mkLinearMap S M) x₁✝) ((LocalizedModule.mkLinearMap …
                           ⊢ Exists fun c => Eq (HSMul.hSMul c x₁✝) (HSMul.hSMul c x₂✝)
                         -/
  exists_of_eq eq1 := by simpa only [eq_comm, one_smul] using LocalizedModule.mk_eq.mp eq1
                         /-
                           🎉 no goals
                         -/


lemma IsLocalizedModule.of_restrictScalars (S : Submonoid R)
    {N : Type*} [AddCommGroup N] [Module R N] [Module A M] [Module A N]
    [IsScalarTower R A M] [IsScalarTower R A N]
    (f : M →ₗ[A] N) [IsLocalizedModule S (f.restrictScalars R)] :
    IsLocalizedModule (Algebra.algebraMapSubmonoid A S) f where
  map_units x := by
    /-
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      M : Type u_2
      inst✝¹⁰ : AddCommMonoid M
      A : Type u_5
      inst✝⁹ : CommSemiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : Module R M
      S : Submonoid R
      N : Type u_6
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : Module R N
      inst✝⁴ : Module A M
      inst✝³ : Module A N
      inst✝² : IsScalarTower R A M
      inst✝¹ : IsScalarTower R A N
      f : LinearMap (RingHom.id A) M N
      inst✝ : IsLocalizedModule S (↑R f)
      x : Subtype fun x => Membership.mem (Algebra.algebraMapSubmonoid A S) x
      ⊢ IsUnit ((algebraMap A (Module.End A N)) ↑x)
    -/
    obtain ⟨_, x, hx, rfl⟩ := x
    /-
      case mk.intro.intro
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      M : Type u_2
      inst✝¹⁰ : AddCommMonoid M
      A : Type u_5
      inst✝⁹ : CommSemiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : Module R M
      S : Submonoid R
      N : Type u_6
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : Module R N
      inst✝⁴ : Module A M
      inst✝³ : Module A N
      inst✝² : IsScalarTower R A M
      inst✝¹ : IsScalarTower R A N
      f : LinearMap (RingHom.id A) M N
      inst✝ : IsLocalizedModule S (↑R f)
      x : R
      hx : Membership.mem (↑S) x
      ⊢ IsUnit ((algebraMap A (Module.End A N)) ↑⟨(algebraMap R A) x, ⋯⟩)
    -/
    have := IsLocalizedModule.map_units (f.restrictScalars R) ⟨x, hx⟩
    /-
      case mk.intro.intro
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      M : Type u_2
      inst✝¹⁰ : AddCommMonoid M
      A : Type u_5
      inst✝⁹ : CommSemiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : Module R M
      S : Submonoid R
      N : Type u_6
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : Module R N
      inst✝⁴ : Module A M
      inst✝³ : Module A N
      inst✝² : IsScalarTower R A M
      inst✝¹ : IsScalarTower R A N
      f : LinearMap (RingHom.id A) M N
      inst✝ : IsLocalizedModule S (↑R f)
      x : R
      hx : Membership.mem (↑S) x
      this : IsUnit ((algebraMap R (Module.End R N)) ↑⟨x, hx⟩)
      ⊢ IsUnit ((algebraMap A (Module.End A N)) ↑⟨(algebraMap R A) x, ⋯⟩)
    -/
    simp only [← IsScalarTower.algebraMap_apply, Module.End_isUnit_iff] at this ⊢
    /-
      case mk.intro.intro
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      M : Type u_2
      inst✝¹⁰ : AddCommMonoid M
      A : Type u_5
      inst✝⁹ : CommSemiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : Module R M
      S : Submonoid R
      N : Type u_6
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : Module R N
      inst✝⁴ : Module A M
      inst✝³ : Module A N
      inst✝² : IsScalarTower R A M
      inst✝¹ : IsScalarTower R A N
      f : LinearMap (RingHom.id A) M N
      inst✝ : IsLocalizedModule S (↑R f)
      x : R
      hx : Membership.mem (↑S) x
      this : Function.Bijective ⇑((algebraMap R (Module.End R N)) x)
      ⊢ Function.Bijective ⇑((algebraMap R (Module.End A N)) x)
    -/
    exact this
    /-
      🎉 no goals
    -/
  surj' y := by
    /-
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      M : Type u_2
      inst✝¹⁰ : AddCommMonoid M
      A : Type u_5
      inst✝⁹ : CommSemiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : Module R M
      S : Submonoid R
      N : Type u_6
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : Module R N
      inst✝⁴ : Module A M
      inst✝³ : Module A N
      inst✝² : IsScalarTower R A M
      inst✝¹ : IsScalarTower R A N
      f : LinearMap (RingHom.id A) M N
      inst✝ : IsLocalizedModule S (↑R f)
      y : N
      ⊢ Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
    -/
    obtain ⟨⟨x, t⟩, e⟩ := IsLocalizedModule.surj S (f.restrictScalars R) y
    /-
      case intro.mk
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      M : Type u_2
      inst✝¹⁰ : AddCommMonoid M
      A : Type u_5
      inst✝⁹ : CommSemiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : Module R M
      S : Submonoid R
      N : Type u_6
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : Module R N
      inst✝⁴ : Module A M
      inst✝³ : Module A N
      inst✝² : IsScalarTower R A M
      inst✝¹ : IsScalarTower R A N
      f : LinearMap (RingHom.id A) M N
      inst✝ : IsLocalizedModule S (↑R f)
      y : N
      x : M
      t : Subtype fun x => Membership.mem S x
      e : Eq (HSMul.hSMul { fst := x, snd := t }.2 y) ((↑R f) { fst := x, snd := t } …
      ⊢ Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
    -/
    exact ⟨⟨x, ⟨_, t, t.2, rfl⟩⟩, by simpa [Submonoid.smul_def] using e⟩
    /-
      🎉 no goals
    -/
  exists_of_eq {x₁ x₂} e := by
    /-
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      M : Type u_2
      inst✝¹⁰ : AddCommMonoid M
      A : Type u_5
      inst✝⁹ : CommSemiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : Module R M
      S : Submonoid R
      N : Type u_6
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : Module R N
      inst✝⁴ : Module A M
      inst✝³ : Module A N
      inst✝² : IsScalarTower R A M
      inst✝¹ : IsScalarTower R A N
      f : LinearMap (RingHom.id A) M N
      inst✝ : IsLocalizedModule S (↑R f)
      x₁ x₂ : M
      e : Eq (f x₁) (f x₂)
      ⊢ Exists fun c => Eq (HSMul.hSMul c x₁) (HSMul.hSMul c x₂)
    -/
    obtain ⟨c, hc⟩ := IsLocalizedModule.exists_of_eq (S := S) (f := f.restrictScalars R) e
    /-
      case intro
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      M : Type u_2
      inst✝¹⁰ : AddCommMonoid M
      A : Type u_5
      inst✝⁹ : CommSemiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : Module R M
      S : Submonoid R
      N : Type u_6
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : Module R N
      inst✝⁴ : Module A M
      inst✝³ : Module A N
      inst✝² : IsScalarTower R A M
      inst✝¹ : IsScalarTower R A N
      f : LinearMap (RingHom.id A) M N
      inst✝ : IsLocalizedModule S (↑R f)
      x₁ x₂ : M
      e : Eq (f x₁) (f x₂)
      c : Subtype fun x => Membership.mem S x
      hc : Eq (HSMul.hSMul c x₁) (HSMul.hSMul c x₂)
      ⊢ Exists fun c => Eq (HSMul.hSMul c x₁) (HSMul.hSMul c x₂)
    -/
    refine ⟨⟨_, c, c.2, rfl⟩, by simpa [Submonoid.smul_def]⟩
    /-
      🎉 no goals
    -/


lemma IsLocalizedModule.of_exists_mul_mem {N : Type*} [AddCommGroup N] [Module R N]
    (S T : Submonoid R) (h : S ≤ T) (h' : ∀ x : T, ∃ m : R, m * x ∈ S)
    (f : M →ₗ[R] N) [IsLocalizedModule S f] :
    IsLocalizedModule T f where
  map_units x := by
    /-
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Type u_2
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      N : Type u_6
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      S T : Submonoid R
      h : LE.le S T
      h' : ∀ (x : Subtype fun x => Membership.mem T x), Exists fun m => Membership.m …
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule S f
      x : Subtype fun x => Membership.mem T x
      ⊢ IsUnit ((algebraMap R (Module.End R N)) ↑x)
    -/
    obtain ⟨m, mx⟩ := h' x
    /-
      case intro
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Type u_2
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      N : Type u_6
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      S T : Submonoid R
      h : LE.le S T
      h' : ∀ (x : Subtype fun x => Membership.mem T x), Exists fun m => Membership.m …
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule S f
      x : Subtype fun x => Membership.mem T x
      m : R
      mx : Membership.mem S (HMul.hMul m ↑x)
      ⊢ IsUnit ((algebraMap R (Module.End R N)) ↑x)
    -/
    have := IsLocalizedModule.map_units f ⟨_, mx⟩
    /-
      case intro
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Type u_2
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      N : Type u_6
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      S T : Submonoid R
      h : LE.le S T
      h' : ∀ (x : Subtype fun x => Membership.mem T x), Exists fun m => Membership.m …
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule S f
      x : Subtype fun x => Membership.mem T x
      m : R
      mx : Membership.mem S (HMul.hMul m ↑x)
      this : IsUnit ((algebraMap R (Module.End R N)) ↑⟨HMul.hMul m ↑x, mx⟩)
      ⊢ IsUnit ((algebraMap R (Module.End R N)) ↑x)
    -/
    rw [map_mul, (Algebra.commute_algebraMap_left _ _).isUnit_mul_iff] at this
    /-
      case intro
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Type u_2
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      N : Type u_6
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      S T : Submonoid R
      h : LE.le S T
      h' : ∀ (x : Subtype fun x => Membership.mem T x), Exists fun m => Membership.m …
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule S f
      x : Subtype fun x => Membership.mem T x
      m : R
      mx : Membership.mem S (HMul.hMul m ↑x)
      this : And (IsUnit ((algebraMap R (Module.End R N)) m)) (IsUnit ((algebraMap R …
      ⊢ IsUnit ((algebraMap R (Module.End R N)) ↑x)
    -/
    exact this.2
    /-
      🎉 no goals
    -/
  surj' y := by
    /-
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Type u_2
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      N : Type u_6
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      S T : Submonoid R
      h : LE.le S T
      h' : ∀ (x : Subtype fun x => Membership.mem T x), Exists fun m => Membership.m …
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule S f
      y : N
      ⊢ Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
    -/
    obtain ⟨⟨x, t⟩, e⟩ := IsLocalizedModule.surj S f y
    /-
      case intro.mk
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Type u_2
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      N : Type u_6
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      S T : Submonoid R
      h : LE.le S T
      h' : ∀ (x : Subtype fun x => Membership.mem T x), Exists fun m => Membership.m …
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule S f
      y : N
      x : M
      t : Subtype fun x => Membership.mem S x
      e : Eq (HSMul.hSMul { fst := x, snd := t }.2 y) (f { fst := x, snd := t }.1)
      ⊢ Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
    -/
    exact ⟨⟨x, ⟨t, h t.2⟩⟩, e⟩
    /-
      🎉 no goals
    -/
  exists_of_eq {x₁ x₂} e := by
    /-
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Type u_2
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      N : Type u_6
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      S T : Submonoid R
      h : LE.le S T
      h' : ∀ (x : Subtype fun x => Membership.mem T x), Exists fun m => Membership.m …
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule S f
      x₁ x₂ : M
      e : Eq (f x₁) (f x₂)
      ⊢ Exists fun c => Eq (HSMul.hSMul c x₁) (HSMul.hSMul c x₂)
    -/
    obtain ⟨c, hc⟩ := IsLocalizedModule.exists_of_eq (S := S) (f := f) e
    /-
      case intro
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Type u_2
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      N : Type u_6
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      S T : Submonoid R
      h : LE.le S T
      h' : ∀ (x : Subtype fun x => Membership.mem T x), Exists fun m => Membership.m …
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule S f
      x₁ x₂ : M
      e : Eq (f x₁) (f x₂)
      c : Subtype fun x => Membership.mem S x
      hc : Eq (HSMul.hSMul c x₁) (HSMul.hSMul c x₂)
      ⊢ Exists fun c => Eq (HSMul.hSMul c x₁) (HSMul.hSMul c x₂)
    -/
    exact ⟨⟨c, h c.2⟩, hc⟩
    /-
      🎉 no goals
    -/


/-- If `(M', f : M ⟶ M')` satisfies universal property of localized module, there is a canonical
map `LocalizedModule S M ⟶ M'`.
-/
noncomputable def fromLocalizedModule' : LocalizedModule S M → M' := fun p =>
  p.liftOn (fun x => (IsLocalizedModule.map_units f x.2).unit⁻¹.val (f x.1))
    (by
      /-
        R : Type u_1
        inst✝¹² : CommSemiring R
        S : Submonoid R
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹¹ : AddCommMonoid M
        inst✝¹⁰ : AddCommMonoid M'
        inst✝⁹ : AddCommMonoid M''
        A : Type u_5
        inst✝⁸ : CommSemiring A
        inst✝⁷ : Algebra R A
        inst✝⁶ : Module A M'
        inst✝⁵ : IsLocalization S A
        inst✝⁴ : Module R M
        inst✝³ : Module R M'
        inst✝² : Module R M''
        inst✝¹ : IsScalarTower R A M'
        f : LinearMap (RingHom.id R) M M'
        g : LinearMap (RingHom.id R) M M''
        inst✝ : IsLocalizedModule S f
        p : LocalizedModule S M
        ⊢ ∀ (p p' : Prod M (Subtype fun x => Membership.mem S x)), HasEquiv.Equiv p p' …
      -/
      rintro ⟨a, b⟩ ⟨a', b'⟩ ⟨c, eq1⟩
      /-
        case mk.mk.intro
        R : Type u_1
        inst✝¹² : CommSemiring R
        S : Submonoid R
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹¹ : AddCommMonoid M
        inst✝¹⁰ : AddCommMonoid M'
        inst✝⁹ : AddCommMonoid M''
        A : Type u_5
        inst✝⁸ : CommSemiring A
        inst✝⁷ : Algebra R A
        inst✝⁶ : Module A M'
        inst✝⁵ : IsLocalization S A
        inst✝⁴ : Module R M
        inst✝³ : Module R M'
        inst✝² : Module R M''
        inst✝¹ : IsScalarTower R A M'
        f : LinearMap (RingHom.id R) M M'
        g : LinearMap (RingHom.id R) M M''
        inst✝ : IsLocalizedModule S f
        p : LocalizedModule S M
        a : M
        b : Subtype fun x => Membership.mem S x
        a' : M
        b' c : Subtype fun x => Membership.mem S x
        eq1 : Eq (HSMul.hSMul c (HSMul.hSMul { fst := a', snd := b' }.2 { fst := a, sn …
        ⊢ Eq ((fun x => ↑(Inv.inv ⋯.unit) (f x.1)) { fst := a, snd := b }) ((fun x =>  …
      -/
      dsimp
      -- Porting note: We remove `generalize_proofs h1 h2`.
      rw [Module.End_algebraMap_isUnit_inv_apply_eq_iff, ← map_smul, ← map_smul,
        Module.End_algebraMap_isUnit_inv_apply_eq_iff', ← map_smul]
      /-
        case mk.mk.intro
        R : Type u_1
        inst✝¹² : CommSemiring R
        S : Submonoid R
        M : Type u_2
        M' : Type u_3
        M'' : Type u_4
        inst✝¹¹ : AddCommMonoid M
        inst✝¹⁰ : AddCommMonoid M'
        inst✝⁹ : AddCommMonoid M''
        A : Type u_5
        inst✝⁸ : CommSemiring A
        inst✝⁷ : Algebra R A
        inst✝⁶ : Module A M'
        inst✝⁵ : IsLocalization S A
        inst✝⁴ : Module R M
        inst✝³ : Module R M'
        inst✝² : Module R M''
        inst✝¹ : IsScalarTower R A M'
        f : LinearMap (RingHom.id R) M M'
        g : LinearMap (RingHom.id R) M M''
        inst✝ : IsLocalizedModule S f
        p : LocalizedModule S M
        a : M
        b : Subtype fun x => Membership.mem S x
        a' : M
        b' c : Subtype fun x => Membership.mem S x
        eq1 : Eq (HSMul.hSMul c (HSMul.hSMul { fst := a', snd := b' }.2 { fst := a, sn …
        ⊢ Eq (f (HSMul.hSMul (↑b) a')) (f (HSMul.hSMul (↑b') a))
      -/
      exact (IsLocalizedModule.eq_iff_exists S f).mpr ⟨c, eq1.symm⟩)
      /-
        🎉 no goals
      -/


@[simp]
theorem fromLocalizedModule'_mk (m : M) (s : S) :
    fromLocalizedModule' S f (LocalizedModule.mk m s) =
      (IsLocalizedModule.map_units f s).unit⁻¹.val (f m) :=
  rfl


theorem fromLocalizedModule'_add (x y : LocalizedModule S M) :
    fromLocalizedModule' S f (x + y) = fromLocalizedModule' S f x + fromLocalizedModule' S f y :=
  LocalizedModule.induction_on₂
    (by
      /-
        R : Type u_1
        inst✝⁵ : CommSemiring R
        S : Submonoid R
        M : Type u_2
        M' : Type u_3
        inst✝⁴ : AddCommMonoid M
        inst✝³ : AddCommMonoid M'
        inst✝² : Module R M
        inst✝¹ : Module R M'
        f : LinearMap (RingHom.id R) M M'
        inst✝ : IsLocalizedModule S f
        x y : LocalizedModule S M
        ⊢ ∀ (m m' : M) (s s' : Subtype fun x => Membership.mem S x), Eq (IsLocalizedMo …
      -/
      intro a a' b b'
      /-
        R : Type u_1
        inst✝⁵ : CommSemiring R
        S : Submonoid R
        M : Type u_2
        M' : Type u_3
        inst✝⁴ : AddCommMonoid M
        inst✝³ : AddCommMonoid M'
        inst✝² : Module R M
        inst✝¹ : Module R M'
        f : LinearMap (RingHom.id R) M M'
        inst✝ : IsLocalizedModule S f
        x y : LocalizedModule S M
        a a' : M
        b b' : Subtype fun x => Membership.mem S x
        ⊢ Eq (IsLocalizedModule.fromLocalizedModule' S f (HAdd.hAdd (LocalizedModule.m …
      -/
      simp only [LocalizedModule.mk_add_mk, fromLocalizedModule'_mk]
      -- Porting note: We remove `generalize_proofs h1 h2 h3`.
      rw [Module.End_algebraMap_isUnit_inv_apply_eq_iff, smul_add, ← map_smul, ← map_smul,
        ← map_smul, map_add]
      /-
        R : Type u_1
        inst✝⁵ : CommSemiring R
        S : Submonoid R
        M : Type u_2
        M' : Type u_3
        inst✝⁴ : AddCommMonoid M
        inst✝³ : AddCommMonoid M'
        inst✝² : Module R M
        inst✝¹ : Module R M'
        f : LinearMap (RingHom.id R) M M'
        inst✝ : IsLocalizedModule S f
        x y : LocalizedModule S M
        a a' : M
        b b' : Subtype fun x => Membership.mem S x
        ⊢ Eq (HAdd.hAdd (f (HSMul.hSMul b' a)) (f (HSMul.hSMul b a'))) (HAdd.hAdd (↑(I …
      -/
      congr 1
      /-
        case e_a
        R : Type u_1
        inst✝⁵ : CommSemiring R
        S : Submonoid R
        M : Type u_2
        M' : Type u_3
        inst✝⁴ : AddCommMonoid M
        inst✝³ : AddCommMonoid M'
        inst✝² : Module R M
        inst✝¹ : Module R M'
        f : LinearMap (RingHom.id R) M M'
        inst✝ : IsLocalizedModule S f
        x y : LocalizedModule S M
        a a' : M
        b b' : Subtype fun x => Membership.mem S x
        ⊢ Eq (f (HSMul.hSMul b' a)) (↑(Inv.inv ⋯.unit) (f (HSMul.hSMul (↑(HMul.hMul b  …
      -/
      all_goals rw [Module.End_algebraMap_isUnit_inv_apply_eq_iff']
        /-
          case e_a
          R : Type u_1
          inst✝⁵ : CommSemiring R
          S : Submonoid R
          M : Type u_2
          M' : Type u_3
          inst✝⁴ : AddCommMonoid M
          inst✝³ : AddCommMonoid M'
          inst✝² : Module R M
          inst✝¹ : Module R M'
          f : LinearMap (RingHom.id R) M M'
          inst✝ : IsLocalizedModule S f
          x y : LocalizedModule S M
          a a' : M
          b b' : Subtype fun x => Membership.mem S x
          ⊢ Eq (f (HSMul.hSMul (↑(HMul.hMul b b')) a)) (HSMul.hSMul (↑b) (f (HSMul.hSMul …
        -/
      · simp [mul_smul, Submonoid.smul_def]
        /-
          🎉 no goals
        -/
        /-
          case e_a
          R : Type u_1
          inst✝⁵ : CommSemiring R
          S : Submonoid R
          M : Type u_2
          M' : Type u_3
          inst✝⁴ : AddCommMonoid M
          inst✝³ : AddCommMonoid M'
          inst✝² : Module R M
          inst✝¹ : Module R M'
          f : LinearMap (RingHom.id R) M M'
          inst✝ : IsLocalizedModule S f
          x y : LocalizedModule S M
          a a' : M
          b b' : Subtype fun x => Membership.mem S x
          ⊢ Eq (HSMul.hSMul (↑(HMul.hMul b b')) (f a')) (HSMul.hSMul (↑b') (f (HSMul.hSM …
        -/
      · rw [Submonoid.coe_mul, LinearMap.map_smul_of_tower, mul_comm, mul_smul, Submonoid.smul_def])
        /-
          🎉 no goals
        -/
    x y


theorem fromLocalizedModule'_smul (r : R) (x : LocalizedModule S M) :
    r • fromLocalizedModule' S f x = fromLocalizedModule' S f (r • x) :=
  LocalizedModule.induction_on
    (by
      /-
        R : Type u_1
        inst✝⁵ : CommSemiring R
        S : Submonoid R
        M : Type u_2
        M' : Type u_3
        inst✝⁴ : AddCommMonoid M
        inst✝³ : AddCommMonoid M'
        inst✝² : Module R M
        inst✝¹ : Module R M'
        f : LinearMap (RingHom.id R) M M'
        inst✝ : IsLocalizedModule S f
        r : R
        x : LocalizedModule S M
        ⊢ ∀ (m : M) (s : Subtype fun x => Membership.mem S x), Eq (HSMul.hSMul r (IsLo …
      -/
      intro a b
      /-
        R : Type u_1
        inst✝⁵ : CommSemiring R
        S : Submonoid R
        M : Type u_2
        M' : Type u_3
        inst✝⁴ : AddCommMonoid M
        inst✝³ : AddCommMonoid M'
        inst✝² : Module R M
        inst✝¹ : Module R M'
        f : LinearMap (RingHom.id R) M M'
        inst✝ : IsLocalizedModule S f
        r : R
        x : LocalizedModule S M
        a : M
        b : Subtype fun x => Membership.mem S x
        ⊢ Eq (HSMul.hSMul r (IsLocalizedModule.fromLocalizedModule' S f (LocalizedModu …
      -/
      rw [fromLocalizedModule'_mk, LocalizedModule.smul'_mk, fromLocalizedModule'_mk]
      -- Porting note: We remove `generalize_proofs h1`.
      /-
        R : Type u_1
        inst✝⁵ : CommSemiring R
        S : Submonoid R
        M : Type u_2
        M' : Type u_3
        inst✝⁴ : AddCommMonoid M
        inst✝³ : AddCommMonoid M'
        inst✝² : Module R M
        inst✝¹ : Module R M'
        f : LinearMap (RingHom.id R) M M'
        inst✝ : IsLocalizedModule S f
        r : R
        x : LocalizedModule S M
        a : M
        b : Subtype fun x => Membership.mem S x
        ⊢ Eq (HSMul.hSMul r (↑(Inv.inv ⋯.unit) (f a))) (↑(Inv.inv ⋯.unit) (f (HSMul.hS …
      -/
      rw [f.map_smul, map_smul])
      /-
        🎉 no goals
      -/
    x


/-- If `(M', f : M ⟶ M')` satisfies universal property of localized module, there is a canonical
map `LocalizedModule S M ⟶ M'`.
-/
noncomputable def fromLocalizedModule : LocalizedModule S M →ₗ[R] M' where
  toFun := fromLocalizedModule' S f
  map_add' := fromLocalizedModule'_add S f
                      /-
                        R : Type u_1
                        inst✝¹² : CommSemiring R
                        S : Submonoid R
                        M : Type u_2
                        M' : Type u_3
                        M'' : Type u_4
                        inst✝¹¹ : AddCommMonoid M
                        inst✝¹⁰ : AddCommMonoid M'
                        inst✝⁹ : AddCommMonoid M''
                        A : Type u_5
                        inst✝⁸ : CommSemiring A
                        inst✝⁷ : Algebra R A
                        inst✝⁶ : Module A M'
                        inst✝⁵ : IsLocalization S A
                        inst✝⁴ : Module R M
                        inst✝³ : Module R M'
                        inst✝² : Module R M''
                        inst✝¹ : IsScalarTower R A M'
                        f : LinearMap (RingHom.id R) M M'
                        g : LinearMap (RingHom.id R) M M''
                        inst✝ : IsLocalizedModule S f
                        r : R
                        x : LocalizedModule S M
                        ⊢ Eq ({ toFun := IsLocalizedModule.fromLocalizedModule' S f, map_add' := ⋯ }.t …
                      -/
  map_smul' r x := by rw [fromLocalizedModule'_smul, RingHom.id_apply]
                      /-
                        🎉 no goals
                      -/


theorem fromLocalizedModule_mk (m : M) (s : S) :
    fromLocalizedModule S f (LocalizedModule.mk m s) =
      (IsLocalizedModule.map_units f s).unit⁻¹.val (f m) :=
  rfl


theorem fromLocalizedModule.inj : Function.Injective <| fromLocalizedModule S f := fun x y eq1 => by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    x y : LocalizedModule S M
    eq1 : Eq ((IsLocalizedModule.fromLocalizedModule S f) x) ((IsLocalizedModule.f …
    ⊢ Eq x y
  -/
  induction' x with a b
  /-
    case h
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    y : LocalizedModule S M
    a : M
    b : Subtype fun x => Membership.mem S x
    eq1 : Eq ((IsLocalizedModule.fromLocalizedModule S f) (LocalizedModule.mk a b) …
    ⊢ Eq (LocalizedModule.mk a b) y
  -/
  induction' y with a' b'
  /-
    case h.h
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    a : M
    b : Subtype fun x => Membership.mem S x
    a' : M
    b' : Subtype fun x => Membership.mem S x
    eq1 : Eq ((IsLocalizedModule.fromLocalizedModule S f) (LocalizedModule.mk a b) …
    ⊢ Eq (LocalizedModule.mk a b) (LocalizedModule.mk a' b')
  -/
  simp only [fromLocalizedModule_mk] at eq1
  -- Porting note: We remove `generalize_proofs h1 h2`.
  rw [Module.End_algebraMap_isUnit_inv_apply_eq_iff, ← LinearMap.map_smul,
    Module.End_algebraMap_isUnit_inv_apply_eq_iff'] at eq1
  rw [LocalizedModule.mk_eq, ← IsLocalizedModule.eq_iff_exists S f, Submonoid.smul_def,
    Submonoid.smul_def, f.map_smul, f.map_smul, eq1]


theorem fromLocalizedModule.surj : Function.Surjective <| fromLocalizedModule S f := fun x =>
  let ⟨⟨m, s⟩, eq1⟩ := IsLocalizedModule.surj S f x
  ⟨LocalizedModule.mk m s, by
    rw [fromLocalizedModule_mk, Module.End_algebraMap_isUnit_inv_apply_eq_iff, ← eq1,
      Submonoid.smul_def]⟩


theorem fromLocalizedModule.bij : Function.Bijective <| fromLocalizedModule S f :=
  ⟨fromLocalizedModule.inj _ _, fromLocalizedModule.surj _ _⟩


/--
If `(M', f : M ⟶ M')` satisfies universal property of localized module, then `M'` is isomorphic to
`LocalizedModule S M` as an `R`-module.
-/
@[simps!]
noncomputable def iso : LocalizedModule S M ≃ₗ[R] M' :=
  { fromLocalizedModule S f,
    Equiv.ofBijective (fromLocalizedModule S f) <| fromLocalizedModule.bij _ _ with }


theorem iso_apply_mk (m : M) (s : S) :
    iso S f (LocalizedModule.mk m s) = (IsLocalizedModule.map_units f s).unit⁻¹.val (f m) :=
  rfl


theorem iso_symm_apply_aux (m : M') :
    (iso S f).symm m =
      LocalizedModule.mk (IsLocalizedModule.surj S f m).choose.1
        (IsLocalizedModule.surj S f m).choose.2 := by
  -- Porting note: We remove `generalize_proofs _ h2`.
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m : M'
    ⊢ Eq ((IsLocalizedModule.iso S f).symm m) (LocalizedModule.mk ⋯.choose.1 ⋯.cho …
  -/
  apply_fun iso S f using LinearEquiv.injective (iso S f)
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m : M'
    ⊢ Eq ((IsLocalizedModule.iso S f) ((IsLocalizedModule.iso S f).symm m)) ((IsLo …
  -/
  rw [LinearEquiv.apply_symm_apply]
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m : M'
    ⊢ Eq m ((IsLocalizedModule.iso S f) (LocalizedModule.mk ⋯.choose.1 ⋯.choose.2))
  -/
  simp only [iso_apply, LinearMap.toFun_eq_coe, fromLocalizedModule_mk]
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m : M'
    ⊢ Eq m (IsLocalizedModule.fromLocalizedModule' S f (LocalizedModule.mk ⋯.choos …
  -/
  erw [Module.End_algebraMap_isUnit_inv_apply_eq_iff', (surj' _).choose_spec]
  /-
    🎉 no goals
  -/


theorem iso_symm_apply' (m : M') (a : M) (b : S) (eq1 : b • m = f a) :
    (iso S f).symm m = LocalizedModule.mk a b :=
  (iso_symm_apply_aux S f m).trans <|
    LocalizedModule.mk_eq.mpr <| by
      -- Porting note: We remove `generalize_proofs h1`.
      rw [← IsLocalizedModule.eq_iff_exists S f, Submonoid.smul_def, Submonoid.smul_def, f.map_smul,
        f.map_smul, ← (surj' _).choose_spec, ← Submonoid.smul_def, ← Submonoid.smul_def, ← mul_smul,
        mul_comm, mul_smul, eq1]


theorem iso_symm_comp : (iso S f).symm.toLinearMap.comp f = LocalizedModule.mkLinearMap S M := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    ⊢ Eq ((↑(IsLocalizedModule.iso S f).symm).comp f) (LocalizedModule.mkLinearMap …
  -/
  ext m
  /-
    case h
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m : M
    ⊢ Eq (((↑(IsLocalizedModule.iso S f).symm).comp f) m) ((LocalizedModule.mkLine …
  -/
  rw [LinearMap.comp_apply, LocalizedModule.mkLinearMap_apply, LinearEquiv.coe_coe, iso_symm_apply']
  /-
    case h.eq1
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m : M
    ⊢ Eq (HSMul.hSMul 1 (f m)) (f m)
  -/
  exact one_smul _ _
  /-
    🎉 no goals
  -/


/--
If `M'` is a localized module and `g` is a linear map `M → M''` such that all scalar multiplication
by `s : S` is invertible, then there is a linear map `M' → M''`.
-/
noncomputable def lift (g : M →ₗ[R] M'')
    (h : ∀ x : S, IsUnit ((algebraMap R (Module.End R M'')) x)) : M' →ₗ[R] M'' :=
  (LocalizedModule.lift S g h).comp (iso S f).symm.toLinearMap


theorem lift_comp (g : M →ₗ[R] M'') (h : ∀ x : S, IsUnit ((algebraMap R (Module.End R M'')) x)) :
    (lift S f g h).comp f = g := by
  /-
    R : Type u_1
    inst✝⁷ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    M'' : Type u_4
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : AddCommMonoid M'
    inst✝⁴ : AddCommMonoid M''
    inst✝³ : Module R M
    inst✝² : Module R M'
    inst✝¹ : Module R M''
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    g : LinearMap (RingHom.id R) M M''
    h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
    ⊢ Eq ((IsLocalizedModule.lift S f g h).comp f) g
  -/
  dsimp only [IsLocalizedModule.lift]
  /-
    R : Type u_1
    inst✝⁷ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    M'' : Type u_4
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : AddCommMonoid M'
    inst✝⁴ : AddCommMonoid M''
    inst✝³ : Module R M
    inst✝² : Module R M'
    inst✝¹ : Module R M''
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    g : LinearMap (RingHom.id R) M M''
    h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
    ⊢ Eq (((LocalizedModule.lift S g h).comp ↑(IsLocalizedModule.iso S f).symm).co …
  -/
  rw [LinearMap.comp_assoc, iso_symm_comp, LocalizedModule.lift_comp S g h]
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_apply (g : M →ₗ[R] M'') (h) (x) :
    lift S f g h (f x) = g x := LinearMap.congr_fun (lift_comp S f g h) x


theorem lift_unique (g : M →ₗ[R] M'') (h : ∀ x : S, IsUnit ((algebraMap R (Module.End R M'')) x))
    (l : M' →ₗ[R] M'') (hl : l.comp f = g) : lift S f g h = l := by
  /-
    R : Type u_1
    inst✝⁷ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    M'' : Type u_4
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : AddCommMonoid M'
    inst✝⁴ : AddCommMonoid M''
    inst✝³ : Module R M
    inst✝² : Module R M'
    inst✝¹ : Module R M''
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    g : LinearMap (RingHom.id R) M M''
    h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
    l : LinearMap (RingHom.id R) M' M''
    hl : Eq (l.comp f) g
    ⊢ Eq (IsLocalizedModule.lift S f g h) l
  -/
  dsimp only [IsLocalizedModule.lift]
  rw [LocalizedModule.lift_unique S g h (l.comp (iso S f).toLinearMap), LinearMap.comp_assoc,
    LinearEquiv.comp_coe, LinearEquiv.symm_trans_self, LinearEquiv.refl_toLinearMap,
    LinearMap.comp_id]
  /-
    R : Type u_1
    inst✝⁷ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    M'' : Type u_4
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : AddCommMonoid M'
    inst✝⁴ : AddCommMonoid M''
    inst✝³ : Module R M
    inst✝² : Module R M'
    inst✝¹ : Module R M''
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    g : LinearMap (RingHom.id R) M M''
    h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
    l : LinearMap (RingHom.id R) M' M''
    hl : Eq (l.comp f) g
    ⊢ Eq ((l.comp ↑(IsLocalizedModule.iso S f)).comp (LocalizedModule.mkLinearMap  …
  -/
  rw [LinearMap.comp_assoc, ← hl]
  /-
    R : Type u_1
    inst✝⁷ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    M'' : Type u_4
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : AddCommMonoid M'
    inst✝⁴ : AddCommMonoid M''
    inst✝³ : Module R M
    inst✝² : Module R M'
    inst✝¹ : Module R M''
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    g : LinearMap (RingHom.id R) M M''
    h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
    l : LinearMap (RingHom.id R) M' M''
    hl : Eq (l.comp f) g
    ⊢ Eq (l.comp ((↑(IsLocalizedModule.iso S f)).comp (LocalizedModule.mkLinearMap …
  -/
  congr 1
  /-
    case e_g
    R : Type u_1
    inst✝⁷ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    M'' : Type u_4
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : AddCommMonoid M'
    inst✝⁴ : AddCommMonoid M''
    inst✝³ : Module R M
    inst✝² : Module R M'
    inst✝¹ : Module R M''
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    g : LinearMap (RingHom.id R) M M''
    h : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R (Module …
    l : LinearMap (RingHom.id R) M' M''
    hl : Eq (l.comp f) g
    ⊢ Eq ((↑(IsLocalizedModule.iso S f)).comp (LocalizedModule.mkLinearMap S M)) f
  -/
  ext x
  rw [LinearMap.comp_apply, LocalizedModule.mkLinearMap_apply, LinearEquiv.coe_coe, iso_apply,
    fromLocalizedModule'_mk, Module.End_algebraMap_isUnit_inv_apply_eq_iff, OneMemClass.coe_one,
    one_smul]


/-- Universal property from localized module:
If `(M', f : M ⟶ M')` is a localized module then it satisfies the following universal property:
For every `R`-module `M''` which every `s : S`-scalar multiplication is invertible and for every
`R`-linear map `g : M ⟶ M''`, there is a unique `R`-linear map `l : M' ⟶ M''` such that
`l ∘ f = g`.
```
M -----f----> M'
|           /
|g       /
|     /   l
v   /
M''
```
-/
theorem is_universal :
    ∀ (g : M →ₗ[R] M'') (_ : ∀ x : S, IsUnit ((algebraMap R (Module.End R M'')) x)),
      ∃! l : M' →ₗ[R] M'', l.comp f = g :=
  fun g h => ⟨lift S f g h, lift_comp S f g h, fun l hl => (lift_unique S f g h l hl).symm⟩


theorem linearMap_ext {N N'} [AddCommMonoid N] [Module R N] [AddCommMonoid N'] [Module R N']
    (f' : N →ₗ[R] N') [IsLocalizedModule S f'] ⦃g g' : M' →ₗ[R] N'⦄
    (h : g ∘ₗ f = g' ∘ₗ f) : g = g' :=
  (is_universal S f _ <| map_units f').unique h rfl


theorem ext (map_unit : ∀ x : S, IsUnit ((algebraMap R (Module.End R M'')) x))
    ⦃j k : M' →ₗ[R] M''⦄ (h : j.comp f = k.comp f) : j = k := by
  /-
    R : Type u_1
    inst✝⁷ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    M'' : Type u_4
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : AddCommMonoid M'
    inst✝⁴ : AddCommMonoid M''
    inst✝³ : Module R M
    inst✝² : Module R M'
    inst✝¹ : Module R M''
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    map_unit : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R  …
    j k : LinearMap (RingHom.id R) M' M''
    h : Eq (j.comp f) (k.comp f)
    ⊢ Eq j k
  -/
  rw [← lift_unique S f (k.comp f) map_unit j h, lift_unique]
  /-
    case hl
    R : Type u_1
    inst✝⁷ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    M'' : Type u_4
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : AddCommMonoid M'
    inst✝⁴ : AddCommMonoid M''
    inst✝³ : Module R M
    inst✝² : Module R M'
    inst✝¹ : Module R M''
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    map_unit : ∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R  …
    j k : LinearMap (RingHom.id R) M' M''
    h : Eq (j.comp f) (k.comp f)
    ⊢ Eq (k.comp f) (k.comp f)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-07")]
alias ringHom_ext := ext


/-- If `(M', f)` and `(M'', g)` both satisfy universal property of localized module, then `M', M''`
are isomorphic as `R`-module
-/
noncomputable def linearEquiv [IsLocalizedModule S g] : M' ≃ₗ[R] M'' :=
  (iso S f).symm.trans (iso S g)


include f in
theorem smul_injective (s : S) : Function.Injective fun m : M' => s • m :=
  ((Module.End_isUnit_iff _).mp (IsLocalizedModule.map_units f s)).injective


include f in
theorem smul_inj (s : S) (m₁ m₂ : M') : s • m₁ = s • m₂ ↔ m₁ = m₂ :=
  (smul_injective f s).eq_iff


/-- `mk' f m s` is the fraction `m/s` with respect to the localization map `f`. -/
noncomputable def mk' (m : M) (s : S) : M' :=
  fromLocalizedModule S f (LocalizedModule.mk m s)


theorem mk'_smul (r : R) (m : M) (s : S) : mk' f (r • m) s = r • mk' f m s := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    r : R
    m : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (IsLocalizedModule.mk' f (HSMul.hSMul r m) s) (HSMul.hSMul r (IsLocalized …
  -/
  delta mk'
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    r : R
    m : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq ((IsLocalizedModule.fromLocalizedModule S f) (LocalizedModule.mk (HSMul.h …
  -/
  rw [← LocalizedModule.smul'_mk, LinearMap.map_smul]
  /-
    🎉 no goals
  -/


theorem mk'_add_mk' (m₁ m₂ : M) (s₁ s₂ : S) :
    mk' f m₁ s₁ + mk' f m₂ s₂ = mk' f (s₂ • m₁ + s₁ • m₂) (s₁ * s₂) := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m₁ m₂ : M
    s₁ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HAdd.hAdd (IsLocalizedModule.mk' f m₁ s₁) (IsLocalizedModule.mk' f m₂ s₂ …
  -/
  delta mk'
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m₁ m₂ : M
    s₁ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HAdd.hAdd ((IsLocalizedModule.fromLocalizedModule S f) (LocalizedModule. …
  -/
  rw [← map_add, LocalizedModule.mk_add_mk]
  /-
    🎉 no goals
  -/


@[simp]
                                               /-
                                                 R : Type u_1
                                                 inst✝⁵ : CommSemiring R
                                                 S : Submonoid R
                                                 M : Type u_2
                                                 M' : Type u_3
                                                 inst✝⁴ : AddCommMonoid M
                                                 inst✝³ : AddCommMonoid M'
                                                 inst✝² : Module R M
                                                 inst✝¹ : Module R M'
                                                 f : LinearMap (RingHom.id R) M M'
                                                 inst✝ : IsLocalizedModule S f
                                                 s : Subtype fun x => Membership.mem S x
                                                 ⊢ Eq (IsLocalizedModule.mk' f 0 s) 0
                                               -/
theorem mk'_zero (s : S) : mk' f 0 s = 0 := by rw [← zero_smul R (0 : M), mk'_smul, zero_smul]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem mk'_one (m : M) : mk' f m (1 : S) = f m := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m : M
    ⊢ Eq (IsLocalizedModule.mk' f m 1) (f m)
  -/
  delta mk'
  rw [fromLocalizedModule_mk, Module.End_algebraMap_isUnit_inv_apply_eq_iff, Submonoid.coe_one,
    one_smul]


@[simp]
theorem mk'_cancel (m : M) (s : S) : mk' f (s • m) s = f m := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (IsLocalizedModule.mk' f (HSMul.hSMul s m) s) (f m)
  -/
  delta mk'
  rw [LocalizedModule.mk_cancel, ← mk'_one S f, fromLocalizedModule_mk,
    Module.End_algebraMap_isUnit_inv_apply_eq_iff, OneMemClass.coe_one, mk'_one, one_smul]


@[simp]
theorem mk'_cancel' (m : M) (s : S) : s • mk' f m s = f m := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul s (IsLocalizedModule.mk' f m s)) (f m)
  -/
  rw [Submonoid.smul_def, ← mk'_smul, ← Submonoid.smul_def, mk'_cancel]
  /-
    🎉 no goals
  -/


@[simp]
theorem mk'_cancel_left (m : M) (s₁ s₂ : S) : mk' f (s₁ • m) (s₁ * s₂) = mk' f m s₂ := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m : M
    s₁ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq (IsLocalizedModule.mk' f (HSMul.hSMul s₁ m) (HMul.hMul s₁ s₂)) (IsLocaliz …
  -/
  delta mk'
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m : M
    s₁ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq ((IsLocalizedModule.fromLocalizedModule S f) (LocalizedModule.mk (HSMul.h …
  -/
  rw [LocalizedModule.mk_cancel_common_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem mk'_cancel_right (m : M) (s₁ s₂ : S) : mk' f (s₂ • m) (s₁ * s₂) = mk' f m s₁ := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m : M
    s₁ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq (IsLocalizedModule.mk' f (HSMul.hSMul s₂ m) (HMul.hMul s₁ s₂)) (IsLocaliz …
  -/
  delta mk'
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m : M
    s₁ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq ((IsLocalizedModule.fromLocalizedModule S f) (LocalizedModule.mk (HSMul.h …
  -/
  rw [LocalizedModule.mk_cancel_common_right]
  /-
    🎉 no goals
  -/


theorem mk'_add (m₁ m₂ : M) (s : S) : mk' f (m₁ + m₂) s = mk' f m₁ s + mk' f m₂ s := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m₁ m₂ : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (IsLocalizedModule.mk' f (HAdd.hAdd m₁ m₂) s) (HAdd.hAdd (IsLocalizedModu …
  -/
  rw [mk'_add_mk', ← smul_add, mk'_cancel_left]
  /-
    🎉 no goals
  -/


theorem mk'_eq_mk'_iff (m₁ m₂ : M) (s₁ s₂ : S) :
    mk' f m₁ s₁ = mk' f m₂ s₂ ↔ ∃ s : S, s • s₁ • m₂ = s • s₂ • m₁ := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m₁ m₂ : M
    s₁ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Iff (Eq (IsLocalizedModule.mk' f m₁ s₁) (IsLocalizedModule.mk' f m₂ s₂)) (Ex …
  -/
  delta mk'
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m₁ m₂ : M
    s₁ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Iff (Eq ((IsLocalizedModule.fromLocalizedModule S f) (LocalizedModule.mk m₁  …
  -/
  rw [(fromLocalizedModule.inj S f).eq_iff, LocalizedModule.mk_eq]
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m₁ m₂ : M
    s₁ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Iff (Exists fun u => Eq (HSMul.hSMul u (HSMul.hSMul s₂ m₁)) (HSMul.hSMul u ( …
  -/
  simp_rw [eq_comm]
  /-
    🎉 no goals
  -/


theorem mk'_neg {M M' : Type*} [AddCommGroup M] [AddCommGroup M'] [Module R M] [Module R M']
    (f : M →ₗ[R] M') [IsLocalizedModule S f] (m : M) (s : S) : mk' f (-m) s = -mk' f m s := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_6
    M' : Type u_7
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (IsLocalizedModule.mk' f (Neg.neg m) s) (Neg.neg (IsLocalizedModule.mk' f …
  -/
  delta mk'
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_6
    M' : Type u_7
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq ((IsLocalizedModule.fromLocalizedModule S f) (LocalizedModule.mk (Neg.neg …
  -/
  rw [LocalizedModule.mk_neg, map_neg]
  /-
    🎉 no goals
  -/


theorem mk'_sub {M M' : Type*} [AddCommGroup M] [AddCommGroup M'] [Module R M] [Module R M']
    (f : M →ₗ[R] M') [IsLocalizedModule S f] (m₁ m₂ : M) (s : S) :
    mk' f (m₁ - m₂) s = mk' f m₁ s - mk' f m₂ s := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_6
    M' : Type u_7
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m₁ m₂ : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (IsLocalizedModule.mk' f (HSub.hSub m₁ m₂) s) (HSub.hSub (IsLocalizedModu …
  -/
  rw [sub_eq_add_neg, sub_eq_add_neg, mk'_add, mk'_neg]
  /-
    🎉 no goals
  -/


theorem mk'_sub_mk' {M M' : Type*} [AddCommGroup M] [AddCommGroup M'] [Module R M] [Module R M']
    (f : M →ₗ[R] M') [IsLocalizedModule S f] (m₁ m₂ : M) (s₁ s₂ : S) :
    mk' f m₁ s₁ - mk' f m₂ s₂ = mk' f (s₂ • m₁ - s₁ • m₂) (s₁ * s₂) := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_6
    M' : Type u_7
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m₁ m₂ : M
    s₁ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSub.hSub (IsLocalizedModule.mk' f m₁ s₁) (IsLocalizedModule.mk' f m₂ s₂ …
  -/
  rw [sub_eq_add_neg, ← mk'_neg, mk'_add_mk', smul_neg, ← sub_eq_add_neg]
  /-
    🎉 no goals
  -/


theorem mk'_mul_mk'_of_map_mul {M M' : Type*} [Semiring M] [Semiring M'] [Module R M]
    [Algebra R M'] (f : M →ₗ[R] M') (hf : ∀ m₁ m₂, f (m₁ * m₂) = f m₁ * f m₂)
    [IsLocalizedModule S f] (m₁ m₂ : M) (s₁ s₂ : S) :
    mk' f m₁ s₁ * mk' f m₂ s₂ = mk' f (m₁ * m₂) (s₁ * s₂) := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_6
    M' : Type u_7
    inst✝⁴ : Semiring M
    inst✝³ : Semiring M'
    inst✝² : Module R M
    inst✝¹ : Algebra R M'
    f : LinearMap (RingHom.id R) M M'
    hf : ∀ (m₁ m₂ : M), Eq (f (HMul.hMul m₁ m₂)) (HMul.hMul (f m₁) (f m₂))
    inst✝ : IsLocalizedModule S f
    m₁ m₂ : M
    s₁ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HMul.hMul (IsLocalizedModule.mk' f m₁ s₁) (IsLocalizedModule.mk' f m₂ s₂ …
  -/
  symm
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_6
    M' : Type u_7
    inst✝⁴ : Semiring M
    inst✝³ : Semiring M'
    inst✝² : Module R M
    inst✝¹ : Algebra R M'
    f : LinearMap (RingHom.id R) M M'
    hf : ∀ (m₁ m₂ : M), Eq (f (HMul.hMul m₁ m₂)) (HMul.hMul (f m₁) (f m₂))
    inst✝ : IsLocalizedModule S f
    m₁ m₂ : M
    s₁ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq (IsLocalizedModule.mk' f (HMul.hMul m₁ m₂) (HMul.hMul s₁ s₂)) (HMul.hMul  …
  -/
  apply (Module.End_algebraMap_isUnit_inv_apply_eq_iff _ _ _ _).mpr
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_6
    M' : Type u_7
    inst✝⁴ : Semiring M
    inst✝³ : Semiring M'
    inst✝² : Module R M
    inst✝¹ : Algebra R M'
    f : LinearMap (RingHom.id R) M M'
    hf : ∀ (m₁ m₂ : M), Eq (f (HMul.hMul m₁ m₂)) (HMul.hMul (f m₁) (f m₂))
    inst✝ : IsLocalizedModule S f
    m₁ m₂ : M
    s₁ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq (f { fst := HMul.hMul m₁ m₂, snd := HMul.hMul s₁ s₂ }.1) (HSMul.hSMul (↑{ …
  -/
  simp_rw [Submonoid.coe_mul, ← smul_eq_mul]
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_6
    M' : Type u_7
    inst✝⁴ : Semiring M
    inst✝³ : Semiring M'
    inst✝² : Module R M
    inst✝¹ : Algebra R M'
    f : LinearMap (RingHom.id R) M M'
    hf : ∀ (m₁ m₂ : M), Eq (f (HMul.hMul m₁ m₂)) (HMul.hMul (f m₁) (f m₂))
    inst✝ : IsLocalizedModule S f
    m₁ m₂ : M
    s₁ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq (f (HSMul.hSMul m₁ m₂)) (HSMul.hSMul (HSMul.hSMul ↑s₁ ↑s₂) (HSMul.hSMul ( …
  -/
  rw [smul_smul_smul_comm, ← mk'_smul, ← mk'_smul]
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_6
    M' : Type u_7
    inst✝⁴ : Semiring M
    inst✝³ : Semiring M'
    inst✝² : Module R M
    inst✝¹ : Algebra R M'
    f : LinearMap (RingHom.id R) M M'
    hf : ∀ (m₁ m₂ : M), Eq (f (HMul.hMul m₁ m₂)) (HMul.hMul (f m₁) (f m₂))
    inst✝ : IsLocalizedModule S f
    m₁ m₂ : M
    s₁ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq (f (HSMul.hSMul m₁ m₂)) (HSMul.hSMul (IsLocalizedModule.mk' f (HSMul.hSMu …
  -/
  simp_rw [← Submonoid.smul_def, mk'_cancel, smul_eq_mul, hf]
  /-
    🎉 no goals
  -/


theorem mk'_mul_mk' {M M' : Type*} [Semiring M] [Semiring M'] [Algebra R M] [Algebra R M']
    (f : M →ₐ[R] M') [IsLocalizedModule S f.toLinearMap] (m₁ m₂ : M) (s₁ s₂ : S) :
    mk' f.toLinearMap m₁ s₁ * mk' f.toLinearMap m₂ s₂ = mk' f.toLinearMap (m₁ * m₂) (s₁ * s₂) :=
  mk'_mul_mk'_of_map_mul f.toLinearMap (map_mul f) m₁ m₂ s₁ s₂


theorem mk'_eq_iff {m : M} {s : S} {m' : M'} : mk' f m s = m' ↔ f m = s • m' := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m : M
    s : Subtype fun x => Membership.mem S x
    m' : M'
    ⊢ Iff (Eq (IsLocalizedModule.mk' f m s) m') (Eq (f m) (HSMul.hSMul s m'))
  -/
  rw [← smul_inj f s, Submonoid.smul_def, ← mk'_smul, ← Submonoid.smul_def, mk'_cancel]
  /-
    🎉 no goals
  -/


@[simp]
                                                                    /-
                                                                      R : Type u_1
                                                                      inst✝⁵ : CommSemiring R
                                                                      S : Submonoid R
                                                                      M : Type u_2
                                                                      M' : Type u_3
                                                                      inst✝⁴ : AddCommMonoid M
                                                                      inst✝³ : AddCommMonoid M'
                                                                      inst✝² : Module R M
                                                                      inst✝¹ : Module R M'
                                                                      f : LinearMap (RingHom.id R) M M'
                                                                      inst✝ : IsLocalizedModule S f
                                                                      m : M
                                                                      s : Subtype fun x => Membership.mem S x
                                                                      ⊢ Iff (Eq (IsLocalizedModule.mk' f m s) 0) (Eq (f m) 0)
                                                                    -/
theorem mk'_eq_zero {m : M} (s : S) : mk' f m s = 0 ↔ f m = 0 := by rw [mk'_eq_iff, smul_zero]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem mk'_eq_zero' {m : M} (s : S) : mk' f m s = 0 ↔ ∃ s' : S, s' • m = 0 := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    m : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Iff (Eq (IsLocalizedModule.mk' f m s) 0) (Exists fun s' => Eq (HSMul.hSMul s …
  -/
  simp_rw [← mk'_zero f (1 : S), mk'_eq_mk'_iff, smul_zero, one_smul, eq_comm]
  /-
    🎉 no goals
  -/


theorem mk_eq_mk' (s : S) (m : M) :
    LocalizedModule.mk m s = mk' (LocalizedModule.mkLinearMap S M) m s := by
  rw [eq_comm, mk'_eq_iff, Submonoid.smul_def, LocalizedModule.smul'_mk, ← Submonoid.smul_def,
    LocalizedModule.mk_cancel, LocalizedModule.mkLinearMap_apply]


variable (A) in
lemma mk'_smul_mk' (x : R) (m : M) (s t : S) :
    IsLocalization.mk' A x s • mk' f m t = mk' f (x • m) (s * t) := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid M'
    A : Type u_5
    inst✝⁷ : CommSemiring A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Module A M'
    inst✝⁴ : IsLocalization S A
    inst✝³ : Module R M
    inst✝² : Module R M'
    inst✝¹ : IsScalarTower R A M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    x : R
    m : M
    s t : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (IsLocalization.mk' A x s) (IsLocalizedModule.mk' f m t)) (I …
  -/
  apply smul_injective f (s * t)
  /-
    case a
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid M'
    A : Type u_5
    inst✝⁷ : CommSemiring A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Module A M'
    inst✝⁴ : IsLocalization S A
    inst✝³ : Module R M
    inst✝² : Module R M'
    inst✝¹ : IsScalarTower R A M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    x : R
    m : M
    s t : Subtype fun x => Membership.mem S x
    ⊢ Eq ((fun m => HSMul.hSMul (HMul.hMul s t) m) (HSMul.hSMul (IsLocalization.mk …
  -/
  conv_lhs => simp only [smul_assoc, mul_smul, smul_comm t]
  /-
    case a
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid M'
    A : Type u_5
    inst✝⁷ : CommSemiring A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Module A M'
    inst✝⁴ : IsLocalization S A
    inst✝³ : Module R M
    inst✝² : Module R M'
    inst✝¹ : IsScalarTower R A M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    x : R
    m : M
    s t : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul s (HSMul.hSMul (IsLocalization.mk' A x s) (HSMul.hSMul t (Is …
  -/
  simp only [mk'_cancel', map_smul, Submonoid.smul_def s]
  /-
    case a
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid M'
    A : Type u_5
    inst✝⁷ : CommSemiring A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Module A M'
    inst✝⁴ : IsLocalization S A
    inst✝³ : Module R M
    inst✝² : Module R M'
    inst✝¹ : IsScalarTower R A M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    x : R
    m : M
    s t : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (↑s) (HSMul.hSMul (IsLocalization.mk' A x s) (f m))) (HSMul. …
  -/
  rw [← smul_assoc, IsLocalization.smul_mk'_self, algebraMap_smul]
  /-
    🎉 no goals
  -/


theorem eq_zero_iff {m : M} : f m = 0 ↔ ∃ s' : S, s' • m = 0 :=
  (mk'_eq_zero (1 : S)).symm.trans (mk'_eq_zero' f _)


theorem mk'_surjective : Function.Surjective (Function.uncurry <| mk' f : M × S → M') := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    ⊢ Function.Surjective (Function.uncurry (IsLocalizedModule.mk' f))
  -/
  intro x
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    x : M'
    ⊢ Exists fun a => Eq (Function.uncurry (IsLocalizedModule.mk' f) a) x
  -/
  obtain ⟨⟨m, s⟩, e : s • x = f m⟩ := IsLocalizedModule.surj S f x
  /-
    case intro.mk
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    x : M'
    m : M
    s : Subtype fun x => Membership.mem S x
    e : Eq (HSMul.hSMul s x) (f m)
    ⊢ Exists fun a => Eq (Function.uncurry (IsLocalizedModule.mk' f) a) x
  -/
  exact ⟨⟨m, s⟩, mk'_eq_iff.mpr e.symm⟩
  /-
    🎉 no goals
  -/


/-- The natural map `Mₛ →ₗ[R] Mₜ` if `s ≤ t` (in `Submonoid R`). -/
noncomputable
def liftOfLE : M₁ →ₗ[R] M₂ :=
  lift S₁ f₁ f₂ fun x ↦ map_units f₂ ⟨x.1, h x.2⟩


/-- The natural map `Mₛ →ₗ[R] Mₜ` if `s ≤ t` (in `Submonoid R`). -/
noncomputable
abbrev _root_.LocalizedModule.liftOfLE : LocalizedModule S₁ M →ₗ[R] LocalizedModule S₂ M :=
  IsLocalizedModule.liftOfLE S₁ S₂ h
    (LocalizedModule.mkLinearMap S₁ M) (LocalizedModule.mkLinearMap S₂ M)


lemma liftOfLE_comp : (liftOfLE S₁ S₂ h f₁ f₂).comp f₁ = f₂ := lift_comp ..


@[simp] lemma liftOfLE_apply (x) : liftOfLE S₁ S₂ h f₁ f₂ (f₁ x) = f₂ x := lift_apply ..


/-- The image of `m/s` under `liftOfLE` is `m/s`. -/
@[simp]
lemma liftOfLE_mk' (m : M) (s : S₁) :
    liftOfLE S₁ S₂ h f₁ f₂ (mk' f₁ m s) = mk' f₂ m ⟨s.1, h s.2⟩ := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_2
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    M₁ : Type u_7
    M₂ : Type u_6
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R M₁
    inst✝² : Module R M₂
    S₁ S₂ : Submonoid R
    h : LE.le S₁ S₂
    f₁ : LinearMap (RingHom.id R) M M₁
    f₂ : LinearMap (RingHom.id R) M M₂
    inst✝¹ : IsLocalizedModule S₁ f₁
    inst✝ : IsLocalizedModule S₂ f₂
    m : M
    s : Subtype fun x => Membership.mem S₁ x
    ⊢ Eq ((IsLocalizedModule.liftOfLE S₁ S₂ h f₁ f₂) (IsLocalizedModule.mk' f₁ m s …
  -/
  apply ((Module.End_isUnit_iff _).mp (map_units f₂ ⟨s, h s.2⟩)).1
  /-
    case a
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_2
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    M₁ : Type u_7
    M₂ : Type u_6
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R M₁
    inst✝² : Module R M₂
    S₁ S₂ : Submonoid R
    h : LE.le S₁ S₂
    f₁ : LinearMap (RingHom.id R) M M₁
    f₂ : LinearMap (RingHom.id R) M M₂
    inst✝¹ : IsLocalizedModule S₁ f₁
    inst✝ : IsLocalizedModule S₂ f₂
    m : M
    s : Subtype fun x => Membership.mem S₁ x
    ⊢ Eq (((algebraMap R (Module.End R M₂)) ↑⟨↑s, ⋯⟩) ((IsLocalizedModule.liftOfLE …
  -/
  simp only [Module.algebraMap_end_apply, ← map_smul, ← Submonoid.smul_def, mk'_cancel']
  /-
    case a
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_2
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    M₁ : Type u_7
    M₂ : Type u_6
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R M₁
    inst✝² : Module R M₂
    S₁ S₂ : Submonoid R
    h : LE.le S₁ S₂
    f₁ : LinearMap (RingHom.id R) M M₁
    f₂ : LinearMap (RingHom.id R) M M₂
    inst✝¹ : IsLocalizedModule S₁ f₁
    inst✝ : IsLocalizedModule S₂ f₂
    m : M
    s : Subtype fun x => Membership.mem S₁ x
    ⊢ Eq ((IsLocalizedModule.liftOfLE S₁ S₂ h f₁ f₂) (f₁ m)) (HSMul.hSMul s (IsLoc …
  -/
  rw [liftOfLE, lift_apply]
  /-
    case a
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Type u_2
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    M₁ : Type u_7
    M₂ : Type u_6
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R M₁
    inst✝² : Module R M₂
    S₁ S₂ : Submonoid R
    h : LE.le S₁ S₂
    f₁ : LinearMap (RingHom.id R) M M₁
    f₂ : LinearMap (RingHom.id R) M M₂
    inst✝¹ : IsLocalizedModule S₁ f₁
    inst✝ : IsLocalizedModule S₂ f₂
    m : M
    s : Subtype fun x => Membership.mem S₁ x
    ⊢ Eq (f₂ m) (HSMul.hSMul s (IsLocalizedModule.mk' f₂ m ⟨↑s, ⋯⟩))
  -/
  exact (mk'_cancel' (S := S₂) f₂ m ⟨s.1, h s.2⟩).symm
  /-
    🎉 no goals
  -/


instance : IsLocalizedModule S₂ (liftOfLE S₁ S₂ h f₁ f₂) where
  map_units := map_units f₂
  surj' y := by
    /-
      R : Type u_1
      inst✝¹⁸ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁷ : AddCommMonoid M
      inst✝¹⁶ : AddCommMonoid M'
      inst✝¹⁵ : AddCommMonoid M''
      A : Type u_5
      inst✝¹⁴ : CommSemiring A
      inst✝¹³ : Algebra R A
      inst✝¹² : Module A M'
      inst✝¹¹ : IsLocalization S A
      inst✝¹⁰ : Module R M
      inst✝⁹ : Module R M'
      inst✝⁸ : Module R M''
      inst✝⁷ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g : LinearMap (RingHom.id R) M M''
      inst✝⁶ : IsLocalizedModule S f
      M₁ : Type u_6
      M₂ : Type u_7
      inst✝⁵ : AddCommMonoid M₁
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : Module R M₁
      inst✝² : Module R M₂
      S₁ S₂ : Submonoid R
      h : LE.le S₁ S₂
      f₁ : LinearMap (RingHom.id R) M M₁
      f₂ : LinearMap (RingHom.id R) M M₂
      inst✝¹ : IsLocalizedModule S₁ f₁
      inst✝ : IsLocalizedModule S₂ f₂
      y : M₂
      ⊢ Exists fun x => Eq (HSMul.hSMul x.2 y) ((IsLocalizedModule.liftOfLE S₁ S₂ h  …
    -/
    obtain ⟨⟨y', s⟩, e⟩ := IsLocalizedModule.surj S₂ f₂ y
    /-
      case intro.mk
      R : Type u_1
      inst✝¹⁸ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁷ : AddCommMonoid M
      inst✝¹⁶ : AddCommMonoid M'
      inst✝¹⁵ : AddCommMonoid M''
      A : Type u_5
      inst✝¹⁴ : CommSemiring A
      inst✝¹³ : Algebra R A
      inst✝¹² : Module A M'
      inst✝¹¹ : IsLocalization S A
      inst✝¹⁰ : Module R M
      inst✝⁹ : Module R M'
      inst✝⁸ : Module R M''
      inst✝⁷ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g : LinearMap (RingHom.id R) M M''
      inst✝⁶ : IsLocalizedModule S f
      M₁ : Type u_6
      M₂ : Type u_7
      inst✝⁵ : AddCommMonoid M₁
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : Module R M₁
      inst✝² : Module R M₂
      S₁ S₂ : Submonoid R
      h : LE.le S₁ S₂
      f₁ : LinearMap (RingHom.id R) M M₁
      f₂ : LinearMap (RingHom.id R) M M₂
      inst✝¹ : IsLocalizedModule S₁ f₁
      inst✝ : IsLocalizedModule S₂ f₂
      y : M₂
      y' : M
      s : Subtype fun x => Membership.mem S₂ x
      e : Eq (HSMul.hSMul { fst := y', snd := s }.2 y) (f₂ { fst := y', snd := s }.1)
      ⊢ Exists fun x => Eq (HSMul.hSMul x.2 y) ((IsLocalizedModule.liftOfLE S₁ S₂ h  …
    -/
    exact ⟨⟨f₁ y', s⟩, by simpa⟩
    /-
      🎉 no goals
    -/
  exists_of_eq := by
    /-
      R : Type u_1
      inst✝¹⁸ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁷ : AddCommMonoid M
      inst✝¹⁶ : AddCommMonoid M'
      inst✝¹⁵ : AddCommMonoid M''
      A : Type u_5
      inst✝¹⁴ : CommSemiring A
      inst✝¹³ : Algebra R A
      inst✝¹² : Module A M'
      inst✝¹¹ : IsLocalization S A
      inst✝¹⁰ : Module R M
      inst✝⁹ : Module R M'
      inst✝⁸ : Module R M''
      inst✝⁷ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g : LinearMap (RingHom.id R) M M''
      inst✝⁶ : IsLocalizedModule S f
      M₁ : Type u_6
      M₂ : Type u_7
      inst✝⁵ : AddCommMonoid M₁
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : Module R M₁
      inst✝² : Module R M₂
      S₁ S₂ : Submonoid R
      h : LE.le S₁ S₂
      f₁ : LinearMap (RingHom.id R) M M₁
      f₂ : LinearMap (RingHom.id R) M M₂
      inst✝¹ : IsLocalizedModule S₁ f₁
      inst✝ : IsLocalizedModule S₂ f₂
      ⊢ ∀ {x₁ x₂ : M₁}, Eq ((IsLocalizedModule.liftOfLE S₁ S₂ h f₁ f₂) x₁) ((IsLocal …
    -/
    intros x₁ x₂ e
    /-
      R : Type u_1
      inst✝¹⁸ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁷ : AddCommMonoid M
      inst✝¹⁶ : AddCommMonoid M'
      inst✝¹⁵ : AddCommMonoid M''
      A : Type u_5
      inst✝¹⁴ : CommSemiring A
      inst✝¹³ : Algebra R A
      inst✝¹² : Module A M'
      inst✝¹¹ : IsLocalization S A
      inst✝¹⁰ : Module R M
      inst✝⁹ : Module R M'
      inst✝⁸ : Module R M''
      inst✝⁷ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g : LinearMap (RingHom.id R) M M''
      inst✝⁶ : IsLocalizedModule S f
      M₁ : Type u_6
      M₂ : Type u_7
      inst✝⁵ : AddCommMonoid M₁
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : Module R M₁
      inst✝² : Module R M₂
      S₁ S₂ : Submonoid R
      h : LE.le S₁ S₂
      f₁ : LinearMap (RingHom.id R) M M₁
      f₂ : LinearMap (RingHom.id R) M M₂
      inst✝¹ : IsLocalizedModule S₁ f₁
      inst✝ : IsLocalizedModule S₂ f₂
      x₁ x₂ : M₁
      e : Eq ((IsLocalizedModule.liftOfLE S₁ S₂ h f₁ f₂) x₁) ((IsLocalizedModule.lif …
      ⊢ Exists fun c => Eq (HSMul.hSMul c x₁) (HSMul.hSMul c x₂)
    -/
    obtain ⟨x₁, s₁, rfl⟩ := mk'_surjective S₁ f₁ x₁
    /-
      case intro.refl
      R : Type u_1
      inst✝¹⁸ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁷ : AddCommMonoid M
      inst✝¹⁶ : AddCommMonoid M'
      inst✝¹⁵ : AddCommMonoid M''
      A : Type u_5
      inst✝¹⁴ : CommSemiring A
      inst✝¹³ : Algebra R A
      inst✝¹² : Module A M'
      inst✝¹¹ : IsLocalization S A
      inst✝¹⁰ : Module R M
      inst✝⁹ : Module R M'
      inst✝⁸ : Module R M''
      inst✝⁷ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g : LinearMap (RingHom.id R) M M''
      inst✝⁶ : IsLocalizedModule S f
      M₁ : Type u_6
      M₂ : Type u_7
      inst✝⁵ : AddCommMonoid M₁
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : Module R M₁
      inst✝² : Module R M₂
      S₁ S₂ : Submonoid R
      h : LE.le S₁ S₂
      f₁ : LinearMap (RingHom.id R) M M₁
      f₂ : LinearMap (RingHom.id R) M M₂
      inst✝¹ : IsLocalizedModule S₁ f₁
      inst✝ : IsLocalizedModule S₂ f₂
      x₂ : M₁
      x₁ : Prod M (Subtype fun x => Membership.mem S₁ x)
      e : Eq ((IsLocalizedModule.liftOfLE S₁ S₂ h f₁ f₂) (Function.uncurry (IsLocali …
      ⊢ Exists fun c => Eq (HSMul.hSMul c (Function.uncurry (IsLocalizedModule.mk' f …
    -/
    obtain ⟨x₂, s₂, rfl⟩ := mk'_surjective S₁ f₁ x₂
    simp only [Function.uncurry, liftOfLE_mk', mk'_eq_mk'_iff, Submonoid.mk_smul,
      Submonoid.smul_def, ← mk'_smul] at e ⊢
    /-
      case intro.refl.intro.refl
      R : Type u_1
      inst✝¹⁸ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁷ : AddCommMonoid M
      inst✝¹⁶ : AddCommMonoid M'
      inst✝¹⁵ : AddCommMonoid M''
      A : Type u_5
      inst✝¹⁴ : CommSemiring A
      inst✝¹³ : Algebra R A
      inst✝¹² : Module A M'
      inst✝¹¹ : IsLocalization S A
      inst✝¹⁰ : Module R M
      inst✝⁹ : Module R M'
      inst✝⁸ : Module R M''
      inst✝⁷ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g : LinearMap (RingHom.id R) M M''
      inst✝⁶ : IsLocalizedModule S f
      M₁ : Type u_6
      M₂ : Type u_7
      inst✝⁵ : AddCommMonoid M₁
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : Module R M₁
      inst✝² : Module R M₂
      S₁ S₂ : Submonoid R
      h : LE.le S₁ S₂
      f₁ : LinearMap (RingHom.id R) M M₁
      f₂ : LinearMap (RingHom.id R) M M₂
      inst✝¹ : IsLocalizedModule S₁ f₁
      inst✝ : IsLocalizedModule S₂ f₂
      x₁ x₂ : Prod M (Subtype fun x => Membership.mem S₁ x)
      e : Exists fun s => Eq (HSMul.hSMul (↑s) (HSMul.hSMul (↑x₁.2) x₂.1)) (HSMul.hS …
      ⊢ Exists fun c => Exists fun s => Eq (HSMul.hSMul (↑s) (HSMul.hSMul (↑x₁.2) (H …
    -/
    obtain ⟨c, e⟩ := e
    /-
      case intro.refl.intro.refl.intro
      R : Type u_1
      inst✝¹⁸ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁷ : AddCommMonoid M
      inst✝¹⁶ : AddCommMonoid M'
      inst✝¹⁵ : AddCommMonoid M''
      A : Type u_5
      inst✝¹⁴ : CommSemiring A
      inst✝¹³ : Algebra R A
      inst✝¹² : Module A M'
      inst✝¹¹ : IsLocalization S A
      inst✝¹⁰ : Module R M
      inst✝⁹ : Module R M'
      inst✝⁸ : Module R M''
      inst✝⁷ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g : LinearMap (RingHom.id R) M M''
      inst✝⁶ : IsLocalizedModule S f
      M₁ : Type u_6
      M₂ : Type u_7
      inst✝⁵ : AddCommMonoid M₁
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : Module R M₁
      inst✝² : Module R M₂
      S₁ S₂ : Submonoid R
      h : LE.le S₁ S₂
      f₁ : LinearMap (RingHom.id R) M M₁
      f₂ : LinearMap (RingHom.id R) M M₂
      inst✝¹ : IsLocalizedModule S₁ f₁
      inst✝ : IsLocalizedModule S₂ f₂
      x₁ x₂ : Prod M (Subtype fun x => Membership.mem S₁ x)
      c : Subtype fun x => Membership.mem S₂ x
      e : Eq (HSMul.hSMul (↑c) (HSMul.hSMul (↑x₁.2) x₂.1)) (HSMul.hSMul (↑c) (HSMul. …
      ⊢ Exists fun c => Exists fun s => Eq (HSMul.hSMul (↑s) (HSMul.hSMul (↑x₁.2) (H …
    -/
    exact ⟨c, 1, by simpa [← smul_comm c.1]⟩
    /-
      🎉 no goals
    -/


/-- A linear map `M →ₗ[R] N` gives a map between localized modules `Mₛ →ₗ[R] Nₛ`. -/
noncomputable
def map : (M →ₗ[R] N) →ₗ[R] (M' →ₗ[R] N') where
  toFun h := lift S f (g ∘ₗ h) (IsLocalizedModule.map_units g)
  map_add' h₁ h₂ := by
    /-
      R : Type u_1
      inst✝¹⁷ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁶ : AddCommMonoid M
      inst✝¹⁵ : AddCommMonoid M'
      inst✝¹⁴ : AddCommMonoid M''
      A : Type u_5
      inst✝¹³ : CommSemiring A
      inst✝¹² : Algebra R A
      inst✝¹¹ : Module A M'
      inst✝¹⁰ : IsLocalization S A
      inst✝⁹ : Module R M
      inst✝⁸ : Module R M'
      inst✝⁷ : Module R M''
      inst✝⁶ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g✝ : LinearMap (RingHom.id R) M M''
      inst✝⁵ : IsLocalizedModule S f
      N : Type ?u.671597
      N' : Type ?u.671600
      inst✝⁴ : AddCommMonoid N
      inst✝³ : AddCommMonoid N'
      inst✝² : Module R N
      inst✝¹ : Module R N'
      g : LinearMap (RingHom.id R) N N'
      inst✝ : IsLocalizedModule S g
      h₁ h₂ : LinearMap (RingHom.id R) M N
      ⊢ Eq ((fun h => IsLocalizedModule.lift S f (g.comp h) ⋯) (HAdd.hAdd h₁ h₂)) (H …
    -/
    apply IsLocalizedModule.ext S f (IsLocalizedModule.map_units g)
    /-
      case h
      R : Type u_1
      inst✝¹⁷ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁶ : AddCommMonoid M
      inst✝¹⁵ : AddCommMonoid M'
      inst✝¹⁴ : AddCommMonoid M''
      A : Type u_5
      inst✝¹³ : CommSemiring A
      inst✝¹² : Algebra R A
      inst✝¹¹ : Module A M'
      inst✝¹⁰ : IsLocalization S A
      inst✝⁹ : Module R M
      inst✝⁸ : Module R M'
      inst✝⁷ : Module R M''
      inst✝⁶ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g✝ : LinearMap (RingHom.id R) M M''
      inst✝⁵ : IsLocalizedModule S f
      N : Type ?u.671597
      N' : Type ?u.671600
      inst✝⁴ : AddCommMonoid N
      inst✝³ : AddCommMonoid N'
      inst✝² : Module R N
      inst✝¹ : Module R N'
      g : LinearMap (RingHom.id R) N N'
      inst✝ : IsLocalizedModule S g
      h₁ h₂ : LinearMap (RingHom.id R) M N
      ⊢ Eq (((fun h => IsLocalizedModule.lift S f (g.comp h) ⋯) (HAdd.hAdd h₁ h₂)).c …
    -/
    simp only [lift_comp, LinearMap.add_comp, LinearMap.comp_add]
    /-
      🎉 no goals
    -/
  map_smul' r h := by
    /-
      R : Type u_1
      inst✝¹⁷ : CommSemiring R
      S : Submonoid R
      M : Type u_2
      M' : Type u_3
      M'' : Type u_4
      inst✝¹⁶ : AddCommMonoid M
      inst✝¹⁵ : AddCommMonoid M'
      inst✝¹⁴ : AddCommMonoid M''
      A : Type u_5
      inst✝¹³ : CommSemiring A
      inst✝¹² : Algebra R A
      inst✝¹¹ : Module A M'
      inst✝¹⁰ : IsLocalization S A
      inst✝⁹ : Module R M
      inst✝⁸ : Module R M'
      inst✝⁷ : Module R M''
      inst✝⁶ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      g✝ : LinearMap (RingHom.id R) M M''
      inst✝⁵ : IsLocalizedModule S f
      N : Type ?u.671597
      N' : Type ?u.671600
      inst✝⁴ : AddCommMonoid N
      inst✝³ : AddCommMonoid N'
      inst✝² : Module R N
      inst✝¹ : Module R N'
      g : LinearMap (RingHom.id R) N N'
      inst✝ : IsLocalizedModule S g
      r : R
      h : LinearMap (RingHom.id R) M N
      ⊢ Eq ({ toFun := fun h => IsLocalizedModule.lift S f (g.comp h) ⋯, map_add' := …
    -/
    apply IsLocalizedModule.ext S f (IsLocalizedModule.map_units g)
    simp only [lift_comp, LinearMap.add_comp, LinearMap.comp_add, LinearMap.smul_comp,
      LinearMap.comp_smul, RingHom.id_apply]


lemma map_comp (h : M →ₗ[R] N) : (map S f g h) ∘ₗ f = g ∘ₗ h :=
  lift_comp S f (g ∘ₗ h) (IsLocalizedModule.map_units g)


@[simp]
lemma map_apply (h : M →ₗ[R] N) (x) : map S f g h (f x) = g (h x) :=
  lift_apply S f (g ∘ₗ h) (IsLocalizedModule.map_units g) x


@[simp]
lemma map_mk' (h : M →ₗ[R] N) (x) (s : S) :
    map S f g h (IsLocalizedModule.mk' f x s) = (IsLocalizedModule.mk' g (h x) s) := by
  simp only [map, lift, LinearMap.coe_mk, AddHom.coe_mk, LinearMap.coe_comp, LinearEquiv.coe_coe,
    Function.comp_apply]
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid M'
    inst✝⁷ : Module R M
    inst✝⁶ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝⁵ : IsLocalizedModule S f
    N : Type u_6
    N' : Type u_7
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid N'
    inst✝² : Module R N
    inst✝¹ : Module R N'
    g : LinearMap (RingHom.id R) N N'
    inst✝ : IsLocalizedModule S g
    h : LinearMap (RingHom.id R) M N
    x : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq ((LocalizedModule.lift S (g.comp h) ⋯) ((IsLocalizedModule.iso S f).symm  …
  -/
  rw [iso_symm_apply' S f (mk' f x s) x s (mk'_cancel' f x s), LocalizedModule.lift_mk]
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid M'
    inst✝⁷ : Module R M
    inst✝⁶ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝⁵ : IsLocalizedModule S f
    N : Type u_6
    N' : Type u_7
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid N'
    inst✝² : Module R N
    inst✝¹ : Module R N'
    g : LinearMap (RingHom.id R) N N'
    inst✝ : IsLocalizedModule S g
    h : LinearMap (RingHom.id R) M N
    x : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (↑(Inv.inv ⋯.unit) ((g.comp h) x)) (IsLocalizedModule.mk' g (h x) s)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma map_id : map S f f (.id ) = .id := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    ⊢ Eq ((IsLocalizedModule.map S f f) LinearMap.id) LinearMap.id
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    x : M'
    ⊢ Eq (((IsLocalizedModule.map S f f) LinearMap.id) x) (LinearMap.id x)
  -/
  obtain ⟨⟨x, s⟩, rfl⟩ := IsLocalizedModule.mk'_surjective S f x
  /-
    case h.intro.mk
    R : Type u_1
    inst✝⁵ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    x : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (((IsLocalizedModule.map S f f) LinearMap.id) (Function.uncurry (IsLocali …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem map_injective (h : M →ₗ[R] N) (h_inj : Function.Injective h) :
    Function.Injective (map S f g h) := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid M'
    inst✝⁷ : Module R M
    inst✝⁶ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝⁵ : IsLocalizedModule S f
    N : Type u_6
    N' : Type u_7
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid N'
    inst✝² : Module R N
    inst✝¹ : Module R N'
    g : LinearMap (RingHom.id R) N N'
    inst✝ : IsLocalizedModule S g
    h : LinearMap (RingHom.id R) M N
    h_inj : Function.Injective ⇑h
    ⊢ Function.Injective ⇑((IsLocalizedModule.map S f g) h)
  -/
  intros x y
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid M'
    inst✝⁷ : Module R M
    inst✝⁶ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝⁵ : IsLocalizedModule S f
    N : Type u_6
    N' : Type u_7
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid N'
    inst✝² : Module R N
    inst✝¹ : Module R N'
    g : LinearMap (RingHom.id R) N N'
    inst✝ : IsLocalizedModule S g
    h : LinearMap (RingHom.id R) M N
    h_inj : Function.Injective ⇑h
    x y : M'
    ⊢ Eq (((IsLocalizedModule.map S f g) h) x) (((IsLocalizedModule.map S f g) h)  …
  -/
  obtain ⟨⟨x, s⟩, rfl⟩ := IsLocalizedModule.mk'_surjective S f x
  /-
    case intro.mk
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid M'
    inst✝⁷ : Module R M
    inst✝⁶ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝⁵ : IsLocalizedModule S f
    N : Type u_6
    N' : Type u_7
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid N'
    inst✝² : Module R N
    inst✝¹ : Module R N'
    g : LinearMap (RingHom.id R) N N'
    inst✝ : IsLocalizedModule S g
    h : LinearMap (RingHom.id R) M N
    h_inj : Function.Injective ⇑h
    y : M'
    x : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (((IsLocalizedModule.map S f g) h) (Function.uncurry (IsLocalizedModule.m …
  -/
  obtain ⟨⟨y, t⟩, rfl⟩ := IsLocalizedModule.mk'_surjective S f y
  simp only [Function.uncurry_apply_pair, map_mk', mk'_eq_mk'_iff, Subtype.exists,
    Submonoid.mk_smul, exists_prop, forall_exists_index, and_imp]
  /-
    case intro.mk.intro.mk
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid M'
    inst✝⁷ : Module R M
    inst✝⁶ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝⁵ : IsLocalizedModule S f
    N : Type u_6
    N' : Type u_7
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid N'
    inst✝² : Module R N
    inst✝¹ : Module R N'
    g : LinearMap (RingHom.id R) N N'
    inst✝ : IsLocalizedModule S g
    h : LinearMap (RingHom.id R) M N
    h_inj : Function.Injective ⇑h
    x : M
    s : Subtype fun x => Membership.mem S x
    y : M
    t : Subtype fun x => Membership.mem S x
    ⊢ ∀ (x_1 : R), Membership.mem S x_1 → Eq (HSMul.hSMul x_1 (HSMul.hSMul s (h y) …
  -/
  intros c hc e
  /-
    case intro.mk.intro.mk
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid M'
    inst✝⁷ : Module R M
    inst✝⁶ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝⁵ : IsLocalizedModule S f
    N : Type u_6
    N' : Type u_7
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid N'
    inst✝² : Module R N
    inst✝¹ : Module R N'
    g : LinearMap (RingHom.id R) N N'
    inst✝ : IsLocalizedModule S g
    h : LinearMap (RingHom.id R) M N
    h_inj : Function.Injective ⇑h
    x : M
    s : Subtype fun x => Membership.mem S x
    y : M
    t : Subtype fun x => Membership.mem S x
    c : R
    hc : Membership.mem S c
    e : Eq (HSMul.hSMul c (HSMul.hSMul s (h y))) (HSMul.hSMul c (HSMul.hSMul t (h  …
    ⊢ Exists fun a => And (Membership.mem S a) (Eq (HSMul.hSMul a (HSMul.hSMul s y …
  -/
  exact ⟨c, hc, h_inj (by simpa)⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem map_surjective (h : M →ₗ[R] N) (h_surj : Function.Surjective h) :
    Function.Surjective (map S f g h) := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid M'
    inst✝⁷ : Module R M
    inst✝⁶ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝⁵ : IsLocalizedModule S f
    N : Type u_6
    N' : Type u_7
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid N'
    inst✝² : Module R N
    inst✝¹ : Module R N'
    g : LinearMap (RingHom.id R) N N'
    inst✝ : IsLocalizedModule S g
    h : LinearMap (RingHom.id R) M N
    h_surj : Function.Surjective ⇑h
    ⊢ Function.Surjective ⇑((IsLocalizedModule.map S f g) h)
  -/
  intros x
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid M'
    inst✝⁷ : Module R M
    inst✝⁶ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝⁵ : IsLocalizedModule S f
    N : Type u_6
    N' : Type u_7
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid N'
    inst✝² : Module R N
    inst✝¹ : Module R N'
    g : LinearMap (RingHom.id R) N N'
    inst✝ : IsLocalizedModule S g
    h : LinearMap (RingHom.id R) M N
    h_surj : Function.Surjective ⇑h
    x : N'
    ⊢ Exists fun a => Eq (((IsLocalizedModule.map S f g) h) a) x
  -/
  obtain ⟨⟨x, s⟩, rfl⟩ := IsLocalizedModule.mk'_surjective S g x
  /-
    case intro.mk
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid M'
    inst✝⁷ : Module R M
    inst✝⁶ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝⁵ : IsLocalizedModule S f
    N : Type u_6
    N' : Type u_7
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid N'
    inst✝² : Module R N
    inst✝¹ : Module R N'
    g : LinearMap (RingHom.id R) N N'
    inst✝ : IsLocalizedModule S g
    h : LinearMap (RingHom.id R) M N
    h_surj : Function.Surjective ⇑h
    x : N
    s : Subtype fun x => Membership.mem S x
    ⊢ Exists fun a => Eq (((IsLocalizedModule.map S f g) h) a) (Function.uncurry ( …
  -/
  obtain ⟨x, rfl⟩ := h_surj x
  /-
    case intro.mk.intro
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    M' : Type u_3
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid M'
    inst✝⁷ : Module R M
    inst✝⁶ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    inst✝⁵ : IsLocalizedModule S f
    N : Type u_6
    N' : Type u_7
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid N'
    inst✝² : Module R N
    inst✝¹ : Module R N'
    g : LinearMap (RingHom.id R) N N'
    inst✝ : IsLocalizedModule S g
    h : LinearMap (RingHom.id R) M N
    h_surj : Function.Surjective ⇑h
    s : Subtype fun x => Membership.mem S x
    x : M
    ⊢ Exists fun a => Eq (((IsLocalizedModule.map S f g) h) a) (Function.uncurry ( …
  -/
  exact ⟨mk' f x s, by simp⟩
  /-
    🎉 no goals
  -/


/-- The linear map `(LocalizedModule S M) → (LocalizedModule S M)` from `iso` is the identity. -/
lemma iso_localizedModule_eq_refl : iso S (mkLinearMap S M) = refl R (LocalizedModule S M) := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ Eq (IsLocalizedModule.iso S (LocalizedModule.mkLinearMap S M)) (LinearEquiv. …
  -/
  let f := mkLinearMap S M
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M (LocalizedModule S M) := LocalizedModule.mkLine …
    ⊢ Eq (IsLocalizedModule.iso S (LocalizedModule.mkLinearMap S M)) (LinearEquiv. …
  -/
  obtain ⟨e, _, univ⟩ := is_universal S f f (map_units f)
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M (LocalizedModule S M) := LocalizedModule.mkLine …
    e : LinearMap (RingHom.id R) (LocalizedModule S M) (LocalizedModule S M)
    left✝ : Eq (e.comp f) f
    univ : ∀ (y : LinearMap (RingHom.id R) (LocalizedModule S M) (LocalizedModule  …
    ⊢ Eq (IsLocalizedModule.iso S (LocalizedModule.mkLinearMap S M)) (LinearEquiv. …
  -/
  rw [← toLinearMap_inj, univ (iso S f) ((eq_toLinearMap_symm_comp f f).1 (iso_symm_comp S f).symm)]
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M (LocalizedModule S M) := LocalizedModule.mkLine …
    e : LinearMap (RingHom.id R) (LocalizedModule S M) (LocalizedModule S M)
    left✝ : Eq (e.comp f) f
    univ : ∀ (y : LinearMap (RingHom.id R) (LocalizedModule S M) (LocalizedModule  …
    ⊢ Eq e ↑(LinearEquiv.refl R (LocalizedModule S M))
  -/
  exact Eq.symm <| univ (refl R (LocalizedModule S M)) (by simp)
  /-
    🎉 no goals
  -/


/-- Formula for `IsLocalizedModule.map` when each localized module is a `LocalizedModule`.-/
lemma map_LocalizedModules (g : M₀ →ₗ[R] M₁) (m : M₀) (s : S) :
    ((map S (mkLinearMap S M₀) (mkLinearMap S M₁)) g)
    (LocalizedModule.mk m s) = LocalizedModule.mk (g m) s := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    S : Submonoid R
    M₀ : Type u_6
    inst✝³ : AddCommMonoid M₀
    inst✝² : Module R M₀
    M₁ : Type u_7
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    g : LinearMap (RingHom.id R) M₀ M₁
    m : M₀
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (((IsLocalizedModule.map S (LocalizedModule.mkLinearMap S M₀) (LocalizedM …
  -/
  have := (iso_apply_mk S (mkLinearMap S M₁) (g m) s).symm
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    S : Submonoid R
    M₀ : Type u_6
    inst✝³ : AddCommMonoid M₀
    inst✝² : Module R M₀
    M₁ : Type u_7
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    g : LinearMap (RingHom.id R) M₀ M₁
    m : M₀
    s : Subtype fun x => Membership.mem S x
    this : Eq (↑(Inv.inv ⋯.unit) ((LocalizedModule.mkLinearMap S M₁) (g m))) ((IsL …
    ⊢ Eq (((IsLocalizedModule.map S (LocalizedModule.mkLinearMap S M₀) (LocalizedM …
  -/
  rw [iso_localizedModule_eq_refl, refl_apply] at this
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    S : Submonoid R
    M₀ : Type u_6
    inst✝³ : AddCommMonoid M₀
    inst✝² : Module R M₀
    M₁ : Type u_7
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    g : LinearMap (RingHom.id R) M₀ M₁
    m : M₀
    s : Subtype fun x => Membership.mem S x
    this : Eq (↑(Inv.inv ⋯.unit) ((LocalizedModule.mkLinearMap S M₁) (g m))) (Loca …
    ⊢ Eq (((IsLocalizedModule.map S (LocalizedModule.mkLinearMap S M₀) (LocalizedM …
  -/
  simpa [map, lift, iso_localizedModule_eq_refl S M₀]
  /-
    🎉 no goals
  -/


lemma map_iso_commute (g : M₀ →ₗ[R] M₁) : (map S f₀ f₁) g ∘ₗ (iso S f₀) =
    (iso S f₁) ∘ₗ (map S (mkLinearMap S M₀) (mkLinearMap S M₁)) g := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M₀ : Type u_6
    M₀' : Type u_9
    inst✝⁹ : AddCommMonoid M₀
    inst✝⁸ : AddCommMonoid M₀'
    inst✝⁷ : Module R M₀
    inst✝⁶ : Module R M₀'
    f₀ : LinearMap (RingHom.id R) M₀ M₀'
    inst✝⁵ : IsLocalizedModule S f₀
    M₁ : Type u_7
    M₁' : Type u_8
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : AddCommMonoid M₁'
    inst✝² : Module R M₁
    inst✝¹ : Module R M₁'
    f₁ : LinearMap (RingHom.id R) M₁ M₁'
    inst✝ : IsLocalizedModule S f₁
    g : LinearMap (RingHom.id R) M₀ M₁
    ⊢ Eq (((IsLocalizedModule.map S f₀ f₁) g).comp ↑(IsLocalizedModule.iso S f₀))  …
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M₀ : Type u_6
    M₀' : Type u_9
    inst✝⁹ : AddCommMonoid M₀
    inst✝⁸ : AddCommMonoid M₀'
    inst✝⁷ : Module R M₀
    inst✝⁶ : Module R M₀'
    f₀ : LinearMap (RingHom.id R) M₀ M₀'
    inst✝⁵ : IsLocalizedModule S f₀
    M₁ : Type u_7
    M₁' : Type u_8
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : AddCommMonoid M₁'
    inst✝² : Module R M₁
    inst✝¹ : Module R M₁'
    f₁ : LinearMap (RingHom.id R) M₁ M₁'
    inst✝ : IsLocalizedModule S f₁
    g : LinearMap (RingHom.id R) M₀ M₁
    x : LocalizedModule S M₀
    ⊢ Eq ((((IsLocalizedModule.map S f₀ f₁) g).comp ↑(IsLocalizedModule.iso S f₀)) …
  -/
  refine induction_on (fun m s ↦ ((Module.End_isUnit_iff _).1 (map_units f₁ s)).1 ?_) x
  /-
    case h
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M₀ : Type u_6
    M₀' : Type u_9
    inst✝⁹ : AddCommMonoid M₀
    inst✝⁸ : AddCommMonoid M₀'
    inst✝⁷ : Module R M₀
    inst✝⁶ : Module R M₀'
    f₀ : LinearMap (RingHom.id R) M₀ M₀'
    inst✝⁵ : IsLocalizedModule S f₀
    M₁ : Type u_7
    M₁' : Type u_8
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : AddCommMonoid M₁'
    inst✝² : Module R M₁
    inst✝¹ : Module R M₁'
    f₁ : LinearMap (RingHom.id R) M₁ M₁'
    inst✝ : IsLocalizedModule S f₁
    g : LinearMap (RingHom.id R) M₀ M₁
    x : LocalizedModule S M₀
    m : M₀
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (((algebraMap R (Module.End R M₁')) ↑s) ((((IsLocalizedModule.map S f₀ f₁ …
  -/
  repeat rw [Module.algebraMap_end_apply, ← CompatibleSMul.map_smul, smul'_mk, ← mk_smul, mk_cancel]
  /-
    case h
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M₀ : Type u_6
    M₀' : Type u_9
    inst✝⁹ : AddCommMonoid M₀
    inst✝⁸ : AddCommMonoid M₀'
    inst✝⁷ : Module R M₀
    inst✝⁶ : Module R M₀'
    f₀ : LinearMap (RingHom.id R) M₀ M₀'
    inst✝⁵ : IsLocalizedModule S f₀
    M₁ : Type u_7
    M₁' : Type u_8
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : AddCommMonoid M₁'
    inst✝² : Module R M₁
    inst✝¹ : Module R M₁'
    f₁ : LinearMap (RingHom.id R) M₁ M₁'
    inst✝ : IsLocalizedModule S f₁
    g : LinearMap (RingHom.id R) M₀ M₁
    x : LocalizedModule S M₀
    m : M₀
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq ((((IsLocalizedModule.map S f₀ f₁) g).comp ↑(IsLocalizedModule.iso S f₀)) …
  -/
  simp -- Can't be combined with next simp. This uses map_apply, which would be preempted by map.
  /-
    case h
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M₀ : Type u_6
    M₀' : Type u_9
    inst✝⁹ : AddCommMonoid M₀
    inst✝⁸ : AddCommMonoid M₀'
    inst✝⁷ : Module R M₀
    inst✝⁶ : Module R M₀'
    f₀ : LinearMap (RingHom.id R) M₀ M₀'
    inst✝⁵ : IsLocalizedModule S f₀
    M₁ : Type u_7
    M₁' : Type u_8
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : AddCommMonoid M₁'
    inst✝² : Module R M₁
    inst✝¹ : Module R M₁'
    f₁ : LinearMap (RingHom.id R) M₁ M₁'
    inst✝ : IsLocalizedModule S f₁
    g : LinearMap (RingHom.id R) M₀ M₁
    x : LocalizedModule S M₀
    m : M₀
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (f₁ (g m)) (IsLocalizedModule.fromLocalizedModule' S f₁ (((IsLocalizedMod …
  -/
  simp [map, lift, iso_localizedModule_eq_refl, lift_mk]
  /-
    🎉 no goals
  -/


/-- Localization of composition is the composition of localization -/
theorem map_comp' (g : M₀ →ₗ[R] M₁) (h : M₁ →ₗ[R] M₂) :
    map S f₀ f₂ (h ∘ₗ g) = map S f₁ f₂ h ∘ₗ map S f₀ f₁ g := by
  /-
    R : Type u_1
    inst✝¹⁵ : CommSemiring R
    S : Submonoid R
    M₀ : Type u_6
    M₀' : Type u_9
    inst✝¹⁴ : AddCommMonoid M₀
    inst✝¹³ : AddCommMonoid M₀'
    inst✝¹² : Module R M₀
    inst✝¹¹ : Module R M₀'
    f₀ : LinearMap (RingHom.id R) M₀ M₀'
    inst✝¹⁰ : IsLocalizedModule S f₀
    M₁ : Type u_7
    M₁' : Type u_11
    inst✝⁹ : AddCommMonoid M₁
    inst✝⁸ : AddCommMonoid M₁'
    inst✝⁷ : Module R M₁
    inst✝⁶ : Module R M₁'
    f₁ : LinearMap (RingHom.id R) M₁ M₁'
    inst✝⁵ : IsLocalizedModule S f₁
    M₂ : Type u_8
    M₂' : Type u_10
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₂'
    inst✝² : Module R M₂
    inst✝¹ : Module R M₂'
    f₂ : LinearMap (RingHom.id R) M₂ M₂'
    inst✝ : IsLocalizedModule S f₂
    g : LinearMap (RingHom.id R) M₀ M₁
    h : LinearMap (RingHom.id R) M₁ M₂
    ⊢ Eq ((IsLocalizedModule.map S f₀ f₂) (h.comp g)) (((IsLocalizedModule.map S f …
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝¹⁵ : CommSemiring R
    S : Submonoid R
    M₀ : Type u_6
    M₀' : Type u_9
    inst✝¹⁴ : AddCommMonoid M₀
    inst✝¹³ : AddCommMonoid M₀'
    inst✝¹² : Module R M₀
    inst✝¹¹ : Module R M₀'
    f₀ : LinearMap (RingHom.id R) M₀ M₀'
    inst✝¹⁰ : IsLocalizedModule S f₀
    M₁ : Type u_7
    M₁' : Type u_11
    inst✝⁹ : AddCommMonoid M₁
    inst✝⁸ : AddCommMonoid M₁'
    inst✝⁷ : Module R M₁
    inst✝⁶ : Module R M₁'
    f₁ : LinearMap (RingHom.id R) M₁ M₁'
    inst✝⁵ : IsLocalizedModule S f₁
    M₂ : Type u_8
    M₂' : Type u_10
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₂'
    inst✝² : Module R M₂
    inst✝¹ : Module R M₂'
    f₂ : LinearMap (RingHom.id R) M₂ M₂'
    inst✝ : IsLocalizedModule S f₂
    g : LinearMap (RingHom.id R) M₀ M₁
    h : LinearMap (RingHom.id R) M₁ M₂
    x : M₀'
    ⊢ Eq (((IsLocalizedModule.map S f₀ f₂) (h.comp g)) x) ((((IsLocalizedModule.ma …
  -/
  obtain ⟨⟨x, s⟩, rfl⟩ := IsLocalizedModule.mk'_surjective S f₀ x
  /-
    case h.intro.mk
    R : Type u_1
    inst✝¹⁵ : CommSemiring R
    S : Submonoid R
    M₀ : Type u_6
    M₀' : Type u_9
    inst✝¹⁴ : AddCommMonoid M₀
    inst✝¹³ : AddCommMonoid M₀'
    inst✝¹² : Module R M₀
    inst✝¹¹ : Module R M₀'
    f₀ : LinearMap (RingHom.id R) M₀ M₀'
    inst✝¹⁰ : IsLocalizedModule S f₀
    M₁ : Type u_7
    M₁' : Type u_11
    inst✝⁹ : AddCommMonoid M₁
    inst✝⁸ : AddCommMonoid M₁'
    inst✝⁷ : Module R M₁
    inst✝⁶ : Module R M₁'
    f₁ : LinearMap (RingHom.id R) M₁ M₁'
    inst✝⁵ : IsLocalizedModule S f₁
    M₂ : Type u_8
    M₂' : Type u_10
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₂'
    inst✝² : Module R M₂
    inst✝¹ : Module R M₂'
    f₂ : LinearMap (RingHom.id R) M₂ M₂'
    inst✝ : IsLocalizedModule S f₂
    g : LinearMap (RingHom.id R) M₀ M₁
    h : LinearMap (RingHom.id R) M₁ M₂
    x : M₀
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (((IsLocalizedModule.map S f₀ f₂) (h.comp g)) (Function.uncurry (IsLocali …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem mkOfAlgebra {R S S' : Type*} [CommRing R] [CommRing S] [CommRing S'] [Algebra R S]
    [Algebra R S'] (M : Submonoid R) (f : S →ₐ[R] S') (h₁ : ∀ x ∈ M, IsUnit (algebraMap R S' x))
    (h₂ : ∀ y, ∃ x : S × M, x.2 • y = f x.1) (h₃ : ∀ x, f x = 0 → ∃ m : M, m • x = 0) :
    IsLocalizedModule M f.toLinearMap := by
  replace h₃ := fun x =>
    Iff.intro (h₃ x) fun ⟨⟨m, hm⟩, e⟩ =>
      (h₁ m hm).mul_left_cancel <| by
        rw [← Algebra.smul_def]
        simpa [Submonoid.smul_def] using f.congr_arg e
  /-
    R : Type u_6
    S : Type u_7
    S' : Type u_8
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : CommRing S'
    inst✝¹ : Algebra R S
    inst✝ : Algebra R S'
    M : Submonoid R
    f : AlgHom R S S'
    h₁ : ∀ (x : R), Membership.mem M x → IsUnit ((algebraMap R S') x)
    h₂ : ∀ (y : S'), Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
    h₃ : ∀ (x : S), Iff (Eq (f x) 0) (Exists fun m => Eq (HSMul.hSMul m x) 0)
    ⊢ IsLocalizedModule M f.toLinearMap
  -/
  constructor
    /-
      case map_units
      R : Type u_6
      S : Type u_7
      S' : Type u_8
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing S'
      inst✝¹ : Algebra R S
      inst✝ : Algebra R S'
      M : Submonoid R
      f : AlgHom R S S'
      h₁ : ∀ (x : R), Membership.mem M x → IsUnit ((algebraMap R S') x)
      h₂ : ∀ (y : S'), Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
      h₃ : ∀ (x : S), Iff (Eq (f x) 0) (Exists fun m => Eq (HSMul.hSMul m x) 0)
      ⊢ ∀ (x : Subtype fun x => Membership.mem M x), IsUnit ((algebraMap R (Module.E …
    -/
  · intro x
    /-
      case map_units
      R : Type u_6
      S : Type u_7
      S' : Type u_8
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing S'
      inst✝¹ : Algebra R S
      inst✝ : Algebra R S'
      M : Submonoid R
      f : AlgHom R S S'
      h₁ : ∀ (x : R), Membership.mem M x → IsUnit ((algebraMap R S') x)
      h₂ : ∀ (y : S'), Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
      h₃ : ∀ (x : S), Iff (Eq (f x) 0) (Exists fun m => Eq (HSMul.hSMul m x) 0)
      x : Subtype fun x => Membership.mem M x
      ⊢ IsUnit ((algebraMap R (Module.End R S')) ↑x)
    -/
    rw [Module.End_isUnit_iff]
    /-
      case map_units
      R : Type u_6
      S : Type u_7
      S' : Type u_8
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing S'
      inst✝¹ : Algebra R S
      inst✝ : Algebra R S'
      M : Submonoid R
      f : AlgHom R S S'
      h₁ : ∀ (x : R), Membership.mem M x → IsUnit ((algebraMap R S') x)
      h₂ : ∀ (y : S'), Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
      h₃ : ∀ (x : S), Iff (Eq (f x) 0) (Exists fun m => Eq (HSMul.hSMul m x) 0)
      x : Subtype fun x => Membership.mem M x
      ⊢ Function.Bijective ⇑((algebraMap R (Module.End R S')) ↑x)
    -/
    constructor
      /-
        case map_units.left
        R : Type u_6
        S : Type u_7
        S' : Type u_8
        inst✝⁴ : CommRing R
        inst✝³ : CommRing S
        inst✝² : CommRing S'
        inst✝¹ : Algebra R S
        inst✝ : Algebra R S'
        M : Submonoid R
        f : AlgHom R S S'
        h₁ : ∀ (x : R), Membership.mem M x → IsUnit ((algebraMap R S') x)
        h₂ : ∀ (y : S'), Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
        h₃ : ∀ (x : S), Iff (Eq (f x) 0) (Exists fun m => Eq (HSMul.hSMul m x) 0)
        x : Subtype fun x => Membership.mem M x
        ⊢ Function.Injective ⇑((algebraMap R (Module.End R S')) ↑x)
      -/
    · rintro a b (e : x • a = x • b)
      /-
        case map_units.left
        R : Type u_6
        S : Type u_7
        S' : Type u_8
        inst✝⁴ : CommRing R
        inst✝³ : CommRing S
        inst✝² : CommRing S'
        inst✝¹ : Algebra R S
        inst✝ : Algebra R S'
        M : Submonoid R
        f : AlgHom R S S'
        h₁ : ∀ (x : R), Membership.mem M x → IsUnit ((algebraMap R S') x)
        h₂ : ∀ (y : S'), Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
        h₃ : ∀ (x : S), Iff (Eq (f x) 0) (Exists fun m => Eq (HSMul.hSMul m x) 0)
        x : Subtype fun x => Membership.mem M x
        a b : S'
        e : Eq (HSMul.hSMul x a) (HSMul.hSMul x b)
        ⊢ Eq a b
      -/
      simp_rw [Submonoid.smul_def, Algebra.smul_def] at e
      /-
        case map_units.left
        R : Type u_6
        S : Type u_7
        S' : Type u_8
        inst✝⁴ : CommRing R
        inst✝³ : CommRing S
        inst✝² : CommRing S'
        inst✝¹ : Algebra R S
        inst✝ : Algebra R S'
        M : Submonoid R
        f : AlgHom R S S'
        h₁ : ∀ (x : R), Membership.mem M x → IsUnit ((algebraMap R S') x)
        h₂ : ∀ (y : S'), Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
        h₃ : ∀ (x : S), Iff (Eq (f x) 0) (Exists fun m => Eq (HSMul.hSMul m x) 0)
        x : Subtype fun x => Membership.mem M x
        a b : S'
        e : Eq (HMul.hMul ((algebraMap R S') ↑x) a) (HMul.hMul ((algebraMap R S') ↑x) b)
        ⊢ Eq a b
      -/
      exact (h₁ x x.2).mul_left_cancel e
      /-
        🎉 no goals
      -/
      /-
        case map_units.right
        R : Type u_6
        S : Type u_7
        S' : Type u_8
        inst✝⁴ : CommRing R
        inst✝³ : CommRing S
        inst✝² : CommRing S'
        inst✝¹ : Algebra R S
        inst✝ : Algebra R S'
        M : Submonoid R
        f : AlgHom R S S'
        h₁ : ∀ (x : R), Membership.mem M x → IsUnit ((algebraMap R S') x)
        h₂ : ∀ (y : S'), Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
        h₃ : ∀ (x : S), Iff (Eq (f x) 0) (Exists fun m => Eq (HSMul.hSMul m x) 0)
        x : Subtype fun x => Membership.mem M x
        ⊢ Function.Surjective ⇑((algebraMap R (Module.End R S')) ↑x)
      -/
    · intro a
      /-
        case map_units.right
        R : Type u_6
        S : Type u_7
        S' : Type u_8
        inst✝⁴ : CommRing R
        inst✝³ : CommRing S
        inst✝² : CommRing S'
        inst✝¹ : Algebra R S
        inst✝ : Algebra R S'
        M : Submonoid R
        f : AlgHom R S S'
        h₁ : ∀ (x : R), Membership.mem M x → IsUnit ((algebraMap R S') x)
        h₂ : ∀ (y : S'), Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
        h₃ : ∀ (x : S), Iff (Eq (f x) 0) (Exists fun m => Eq (HSMul.hSMul m x) 0)
        x : Subtype fun x => Membership.mem M x
        a : S'
        ⊢ Exists fun a_1 => Eq (((algebraMap R (Module.End R S')) ↑x) a_1) a
      -/
      refine ⟨((h₁ x x.2).unit⁻¹ : _) * a, ?_⟩
      /-
        case map_units.right
        R : Type u_6
        S : Type u_7
        S' : Type u_8
        inst✝⁴ : CommRing R
        inst✝³ : CommRing S
        inst✝² : CommRing S'
        inst✝¹ : Algebra R S
        inst✝ : Algebra R S'
        M : Submonoid R
        f : AlgHom R S S'
        h₁ : ∀ (x : R), Membership.mem M x → IsUnit ((algebraMap R S') x)
        h₂ : ∀ (y : S'), Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
        h₃ : ∀ (x : S), Iff (Eq (f x) 0) (Exists fun m => Eq (HSMul.hSMul m x) 0)
        x : Subtype fun x => Membership.mem M x
        a : S'
        ⊢ Eq (((algebraMap R (Module.End R S')) ↑x) (HMul.hMul (↑(Inv.inv ⋯.unit)) a)) a
      -/
      rw [Module.algebraMap_end_apply, Algebra.smul_def, ← mul_assoc, IsUnit.mul_val_inv, one_mul]
      /-
        🎉 no goals
      -/
    /-
      case surj'
      R : Type u_6
      S : Type u_7
      S' : Type u_8
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing S'
      inst✝¹ : Algebra R S
      inst✝ : Algebra R S'
      M : Submonoid R
      f : AlgHom R S S'
      h₁ : ∀ (x : R), Membership.mem M x → IsUnit ((algebraMap R S') x)
      h₂ : ∀ (y : S'), Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
      h₃ : ∀ (x : S), Iff (Eq (f x) 0) (Exists fun m => Eq (HSMul.hSMul m x) 0)
      ⊢ ∀ (y : S'), Exists fun x => Eq (HSMul.hSMul x.2 y) (f.toLinearMap x.1)
    -/
  · exact h₂
    /-
      🎉 no goals
    -/
    /-
      case exists_of_eq
      R : Type u_6
      S : Type u_7
      S' : Type u_8
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing S'
      inst✝¹ : Algebra R S
      inst✝ : Algebra R S'
      M : Submonoid R
      f : AlgHom R S S'
      h₁ : ∀ (x : R), Membership.mem M x → IsUnit ((algebraMap R S') x)
      h₂ : ∀ (y : S'), Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
      h₃ : ∀ (x : S), Iff (Eq (f x) 0) (Exists fun m => Eq (HSMul.hSMul m x) 0)
      ⊢ ∀ {x₁ x₂ : S}, Eq (f.toLinearMap x₁) (f.toLinearMap x₂) → Exists fun c => Eq …
    -/
  · intros x y
    /-
      case exists_of_eq
      R : Type u_6
      S : Type u_7
      S' : Type u_8
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing S'
      inst✝¹ : Algebra R S
      inst✝ : Algebra R S'
      M : Submonoid R
      f : AlgHom R S S'
      h₁ : ∀ (x : R), Membership.mem M x → IsUnit ((algebraMap R S') x)
      h₂ : ∀ (y : S'), Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
      h₃ : ∀ (x : S), Iff (Eq (f x) 0) (Exists fun m => Eq (HSMul.hSMul m x) 0)
      x y : S
      ⊢ Eq (f.toLinearMap x) (f.toLinearMap y) → Exists fun c => Eq (HSMul.hSMul c x …
    -/
    dsimp only [AlgHom.toLinearMap_apply]
    /-
      case exists_of_eq
      R : Type u_6
      S : Type u_7
      S' : Type u_8
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing S'
      inst✝¹ : Algebra R S
      inst✝ : Algebra R S'
      M : Submonoid R
      f : AlgHom R S S'
      h₁ : ∀ (x : R), Membership.mem M x → IsUnit ((algebraMap R S') x)
      h₂ : ∀ (y : S'), Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
      h₃ : ∀ (x : S), Iff (Eq (f x) 0) (Exists fun m => Eq (HSMul.hSMul m x) 0)
      x y : S
      ⊢ Eq (f x) (f y) → Exists fun c => Eq (HSMul.hSMul c x) (HSMul.hSMul c y)
    -/
    rw [← sub_eq_zero, ← map_sub, h₃]
    /-
      case exists_of_eq
      R : Type u_6
      S : Type u_7
      S' : Type u_8
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing S'
      inst✝¹ : Algebra R S
      inst✝ : Algebra R S'
      M : Submonoid R
      f : AlgHom R S S'
      h₁ : ∀ (x : R), Membership.mem M x → IsUnit ((algebraMap R S') x)
      h₂ : ∀ (y : S'), Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
      h₃ : ∀ (x : S), Iff (Eq (f x) 0) (Exists fun m => Eq (HSMul.hSMul m x) 0)
      x y : S
      ⊢ (Exists fun m => Eq (HSMul.hSMul m (HSub.hSub x y)) 0) → Exists fun c => Eq  …
    -/
    simp_rw [smul_sub, sub_eq_zero]
    /-
      case exists_of_eq
      R : Type u_6
      S : Type u_7
      S' : Type u_8
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing S'
      inst✝¹ : Algebra R S
      inst✝ : Algebra R S'
      M : Submonoid R
      f : AlgHom R S S'
      h₁ : ∀ (x : R), Membership.mem M x → IsUnit ((algebraMap R S') x)
      h₂ : ∀ (y : S'), Exists fun x => Eq (HSMul.hSMul x.2 y) (f x.1)
      h₃ : ∀ (x : S), Iff (Eq (f x) 0) (Exists fun m => Eq (HSMul.hSMul m x) 0)
      x y : S
      ⊢ (Exists fun m => Eq (HSMul.hSMul m x) (HSMul.hSMul m y)) → Exists fun c => E …
    -/
    exact id
    /-
      🎉 no goals
    -/


lemma LocalizedModule.mem_ker_mkLinearMap_iff {S : Submonoid R} {m} :
    m ∈ LinearMap.ker (LocalizedModule.mkLinearMap S M) ↔ ∃ r ∈ S, r • m = 0 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    S : Submonoid R
    m : M
    ⊢ Iff (Membership.mem (LinearMap.ker (LocalizedModule.mkLinearMap S M)) m) (Ex …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      S : Submonoid R
      m : M
      ⊢ Membership.mem (LinearMap.ker (LocalizedModule.mkLinearMap S M)) m → Exists  …
    -/
  · intro H
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      S : Submonoid R
      m : M
      H : Membership.mem (LinearMap.ker (LocalizedModule.mkLinearMap S M)) m
      ⊢ Exists fun r => And (Membership.mem S r) (Eq (HSMul.hSMul r m) 0)
    -/
    obtain ⟨r, hr⟩ := (@LocalizedModule.mk_eq _ _ S M _ _ m 0 1 1).mp (by simpa using H)
    /-
      case mp.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      S : Submonoid R
      m : M
      H : Membership.mem (LinearMap.ker (LocalizedModule.mkLinearMap S M)) m
      r : Subtype fun x => Membership.mem S x
      hr : Eq (HSMul.hSMul r (HSMul.hSMul 1 m)) (HSMul.hSMul r (HSMul.hSMul 1 0))
      ⊢ Exists fun r => And (Membership.mem S r) (Eq (HSMul.hSMul r m) 0)
    -/
    exact ⟨r, r.2, by simpa using hr⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      S : Submonoid R
      m : M
      ⊢ (Exists fun r => And (Membership.mem S r) (Eq (HSMul.hSMul r m) 0)) → Member …
    -/
  · rintro ⟨r, hr, e⟩
    apply ((Module.End_isUnit_iff _).mp
      (IsLocalizedModule.map_units (LocalizedModule.mkLinearMap S M) ⟨r, hr⟩)).injective
    /-
      case mpr.intro.intro.a
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      S : Submonoid R
      m : M
      r : R
      hr : Membership.mem S r
      e : Eq (HSMul.hSMul r m) 0
      ⊢ Eq (((algebraMap R (Module.End R (LocalizedModule S M))) ↑⟨r, hr⟩) ((Localiz …
    -/
    simp [← IsLocalizedModule.mk_eq_mk', LocalizedModule.smul'_mk, e]
    /-
      🎉 no goals
    -/


lemma LocalizedModule.subsingleton_iff_ker_eq_top {S : Submonoid R} :
    Subsingleton (LocalizedModule S M) ↔
      LinearMap.ker (LocalizedModule.mkLinearMap S M) = ⊤ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    S : Submonoid R
    ⊢ Iff (Subsingleton (LocalizedModule S M)) (Eq (LinearMap.ker (LocalizedModule …
  -/
  rw [← top_le_iff]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    S : Submonoid R
    ⊢ Iff (Subsingleton (LocalizedModule S M)) (LE.le Top.top (LinearMap.ker (Loca …
  -/
  refine ⟨fun H m _ ↦ Subsingleton.elim _ _, fun H ↦ (subsingleton_iff_forall_eq 0).mpr fun x ↦ ?_⟩
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    S : Submonoid R
    H : LE.le Top.top (LinearMap.ker (LocalizedModule.mkLinearMap S M))
    x : LocalizedModule S M
    ⊢ Eq x 0
  -/
  obtain ⟨⟨x, s⟩, rfl⟩ := IsLocalizedModule.mk'_surjective S (LocalizedModule.mkLinearMap S M) x
  /-
    case intro.mk
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    S : Submonoid R
    H : LE.le Top.top (LinearMap.ker (LocalizedModule.mkLinearMap S M))
    x : M
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (Function.uncurry (IsLocalizedModule.mk' (LocalizedModule.mkLinearMap S M …
  -/
  simpa using @H x trivial
  /-
    🎉 no goals
  -/


lemma LocalizedModule.subsingleton_iff {S : Submonoid R} :
    Subsingleton (LocalizedModule S M) ↔ ∀ m : M, ∃ r ∈ S, r • m = 0 := by
  simp_rw [subsingleton_iff_ker_eq_top, ← top_le_iff, SetLike.le_def,
    mem_ker_mkLinearMap_iff, Submodule.mem_top, true_implies]


