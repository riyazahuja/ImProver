/-- The commutator of derivations is again a derivation. -/
instance : Bracket (Derivation R A A) (Derivation R A A) :=
  ⟨fun D1 D2 =>
    mk' ⁅(D1 : Module.End R A), (D2 : Module.End R A)⁆ fun a b => by
      simp only [Ring.lie_def, map_add, Algebra.id.smul_eq_mul, LinearMap.mul_apply, leibniz,
        coeFn_coe, LinearMap.sub_apply]
      /-
        R : Type u_1
        inst✝² : CommRing R
        A : Type u_2
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        D1✝ D2✝ : Derivation R A A
        a✝ : A
        D1 D2 : Derivation R A A
        a b : A
        ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (D1 (D2 b))) (HMul.hMul (D2 …
      -/
      ring⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem commutator_coe_linear_map : ↑⁅D1, D2⁆ = ⁅(D1 : Module.End R A), (D2 : Module.End R A)⁆ :=
  rfl


theorem commutator_apply : ⁅D1, D2⁆ a = D1 (D2 a) - D2 (D1 a) :=
  rfl


instance : LieRing (Derivation R A A) where
                      /-
                        R : Type u_1
                        inst✝² : CommRing R
                        A : Type u_2
                        inst✝¹ : CommRing A
                        inst✝ : Algebra R A
                        D1 D2 : Derivation R A A
                        a : A
                        d e f : Derivation R A A
                        ⊢ Eq (Bracket.bracket (HAdd.hAdd d e) f) (HAdd.hAdd (Bracket.bracket d f) (Bra …
                      -/
  add_lie d e f := by ext a; simp only [commutator_apply, add_apply, map_add]; ring
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
                      /-
                        R : Type u_1
                        inst✝² : CommRing R
                        A : Type u_2
                        inst✝¹ : CommRing A
                        inst✝ : Algebra R A
                        D1 D2 : Derivation R A A
                        a : A
                        d e f : Derivation R A A
                        ⊢ Eq (Bracket.bracket d (HAdd.hAdd e f)) (HAdd.hAdd (Bracket.bracket d e) (Bra …
                      -/
  lie_add d e f := by ext a; simp only [commutator_apply, add_apply, map_add]; ring
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
                   /-
                     R : Type u_1
                     inst✝² : CommRing R
                     A : Type u_2
                     inst✝¹ : CommRing A
                     inst✝ : Algebra R A
                     D1 D2 : Derivation R A A
                     a : A
                     d : Derivation R A A
                     ⊢ Eq (Bracket.bracket d d) 0
                   -/
  lie_self d := by ext a; simp only [commutator_apply, add_apply, map_add]; ring_nf; simp
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
                          /-
                            R : Type u_1
                            inst✝² : CommRing R
                            A : Type u_2
                            inst✝¹ : CommRing A
                            inst✝ : Algebra R A
                            D1 D2 : Derivation R A A
                            a : A
                            d e f : Derivation R A A
                            ⊢ Eq (Bracket.bracket d (Bracket.bracket e f)) (HAdd.hAdd (Bracket.bracket (Br …
                          -/
  leibniz_lie d e f := by ext a; simp only [commutator_apply, add_apply, sub_apply, map_sub]; ring
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


instance instLieAlgebra : LieAlgebra R (Derivation R A A) :=
  { Derivation.instModule with
    lie_smul := fun r d e => by
      /-
        R : Type u_1
        inst✝² : CommRing R
        A : Type u_2
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        D1 D2 : Derivation R A A
        a : A
        r : R
        d e : Derivation R A A
        ⊢ Eq (Bracket.bracket d (HSMul.hSMul r e)) (HSMul.hSMul r (Bracket.bracket d e))
      -/
      ext a; simp only [commutator_apply, map_smul, smul_sub, smul_apply] }
             /-
               🎉 no goals
             -/


