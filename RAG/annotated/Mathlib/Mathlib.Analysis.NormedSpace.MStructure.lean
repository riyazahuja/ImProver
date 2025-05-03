/-- A projection on a normed space `X` is said to be an L-projection if, for all `x` in `X`,
$\|x\| = \|P x\| + \|(1 - P) x\|$.

Note that we write `P • x` instead of `P x` for reasons described in the module docstring.
-/
structure IsLprojection (P : M) : Prop where
  proj : IsIdempotentElem P
  Lnorm : ∀ x : X, ‖x‖ = ‖P • x‖ + ‖(1 - P) • x‖


/-- A projection on a normed space `X` is said to be an M-projection if, for all `x` in `X`,
$\|x\| = max(\|P x\|,\|(1 - P) x\|)$.

Note that we write `P • x` instead of `P x` for reasons described in the module docstring.
-/
structure IsMprojection (P : M) : Prop where
  proj : IsIdempotentElem P
  Mnorm : ∀ x : X, ‖x‖ = max ‖P • x‖ ‖(1 - P) • x‖


theorem Lcomplement {P : M} (h : IsLprojection X P) : IsLprojection X (1 - P) :=
  ⟨h.proj.one_sub, fun x => by
    /-
      X : Type u_1
      inst✝² : NormedAddCommGroup X
      M : Type u_2
      inst✝¹ : Ring M
      inst✝ : Module M X
      P : M
      h : IsLprojection X P
      x : X
      ⊢ Eq (Norm.norm x) (HAdd.hAdd (Norm.norm (HSMul.hSMul (HSub.hSub 1 P) x)) (Nor …
    -/
    rw [add_comm, sub_sub_cancel]
    /-
      X : Type u_1
      inst✝² : NormedAddCommGroup X
      M : Type u_2
      inst✝¹ : Ring M
      inst✝ : Module M X
      P : M
      h : IsLprojection X P
      x : X
      ⊢ Eq (Norm.norm x) (HAdd.hAdd (Norm.norm (HSMul.hSMul P x)) (Norm.norm (HSMul. …
    -/
    exact h.Lnorm x⟩
    /-
      🎉 no goals
    -/


theorem Lcomplement_iff (P : M) : IsLprojection X P ↔ IsLprojection X (1 - P) :=
  ⟨Lcomplement, fun h => sub_sub_cancel 1 P ▸ h.Lcomplement⟩


theorem commute [FaithfulSMul M X] {P Q : M} (h₁ : IsLprojection X P) (h₂ : IsLprojection X Q) :
    Commute P Q := by
  have PR_eq_RPR : ∀ R : M, IsLprojection X R → P * R = R * P * R := fun R h₃ => by
    -- Porting note: Needed to fix function, which changes indent of following lines
    refine @eq_of_smul_eq_smul _ X _ _ _ _ fun x => by
      rw [← norm_sub_eq_zero_iff]
      have e1 : ‖R • x‖ ≥ ‖R • x‖ + 2 • ‖(P * R) • x - (R * P * R) • x‖ :=
        calc
          ‖R • x‖ = ‖R • P • R • x‖ + ‖(1 - R) • P • R • x‖ +
              (‖(R * R) • x - R • P • R • x‖ + ‖(1 - R) • (1 - P) • R • x‖) := by
            rw [h₁.Lnorm, h₃.Lnorm, h₃.Lnorm ((1 - P) • R • x), sub_smul 1 P, one_smul, smul_sub,
              mul_smul]
          _ = ‖R • P • R • x‖ + ‖(1 - R) • P • R • x‖ +
              (‖R • x - R • P • R • x‖ + ‖((1 - R) * R) • x - (1 - R) • P • R • x‖) := by
            rw [h₃.proj.eq, sub_smul 1 P, one_smul, smul_sub, mul_smul]
          _ = ‖R • P • R • x‖ + ‖(1 - R) • P • R • x‖ +
              (‖R • x - R • P • R • x‖ + ‖(1 - R) • P • R • x‖) := by
            rw [sub_mul, h₃.proj.eq, one_mul, sub_self, zero_smul, zero_sub, norm_neg]
          _ = ‖R • P • R • x‖ + ‖R • x - R • P • R • x‖ + 2 • ‖(1 - R) • P • R • x‖ := by abel
          _ ≥ ‖R • x‖ + 2 • ‖(P * R) • x - (R * P * R) • x‖ := by
            rw [GE.ge]
            have :=
              add_le_add_right (norm_le_insert' (R • x) (R • P • R • x)) (2 • ‖(1 - R) • P • R • x‖)
            simpa only [mul_smul, sub_smul, one_smul] using this

      rw [GE.ge] at e1
      -- Porting note: Bump index in nth_rewrite
      nth_rewrite 2 [← add_zero ‖R • x‖] at e1
      rw [add_le_add_iff_left, two_smul, ← two_mul] at e1
      rw [le_antisymm_iff]
      refine ⟨?_, norm_nonneg _⟩
      rwa [← mul_zero (2 : ℝ), mul_le_mul_left (show (0 : ℝ) < 2 by norm_num)] at e1
  have QP_eq_QPQ : Q * P = Q * P * Q := by
    have e1 : P * (1 - Q) = P * (1 - Q) - (Q * P - Q * P * Q) :=
      calc
        P * (1 - Q) = (1 - Q) * P * (1 - Q) := by rw [PR_eq_RPR (1 - Q) h₂.Lcomplement]
        _ = P * (1 - Q) - (Q * P - Q * P * Q) := by noncomm_ring
    rwa [eq_sub_iff_add_eq, add_right_eq_self, sub_eq_zero] at e1
  /-
    X : Type u_1
    inst✝³ : NormedAddCommGroup X
    M : Type u_2
    inst✝² : Ring M
    inst✝¹ : Module M X
    inst✝ : FaithfulSMul M X
    P Q : M
    h₁ : IsLprojection X P
    h₂ : IsLprojection X Q
    PR_eq_RPR : ∀ (R : M), IsLprojection X R → Eq (HMul.hMul P R) (HMul.hMul (HMul …
    QP_eq_QPQ : Eq (HMul.hMul Q P) (HMul.hMul (HMul.hMul Q P) Q)
    ⊢ Commute P Q
  -/
  show P * Q = Q * P
  /-
    X : Type u_1
    inst✝³ : NormedAddCommGroup X
    M : Type u_2
    inst✝² : Ring M
    inst✝¹ : Module M X
    inst✝ : FaithfulSMul M X
    P Q : M
    h₁ : IsLprojection X P
    h₂ : IsLprojection X Q
    PR_eq_RPR : ∀ (R : M), IsLprojection X R → Eq (HMul.hMul P R) (HMul.hMul (HMul …
    QP_eq_QPQ : Eq (HMul.hMul Q P) (HMul.hMul (HMul.hMul Q P) Q)
    ⊢ Eq (HMul.hMul P Q) (HMul.hMul Q P)
  -/
  rw [QP_eq_QPQ, PR_eq_RPR Q h₂]
  /-
    🎉 no goals
  -/


theorem mul [FaithfulSMul M X] {P Q : M} (h₁ : IsLprojection X P) (h₂ : IsLprojection X Q) :
    IsLprojection X (P * Q) := by
  /-
    X : Type u_1
    inst✝³ : NormedAddCommGroup X
    M : Type u_2
    inst✝² : Ring M
    inst✝¹ : Module M X
    inst✝ : FaithfulSMul M X
    P Q : M
    h₁ : IsLprojection X P
    h₂ : IsLprojection X Q
    ⊢ IsLprojection X (HMul.hMul P Q)
  -/
  refine ⟨IsIdempotentElem.mul_of_commute (h₁.commute h₂) h₁.proj h₂.proj, ?_⟩
  /-
    X : Type u_1
    inst✝³ : NormedAddCommGroup X
    M : Type u_2
    inst✝² : Ring M
    inst✝¹ : Module M X
    inst✝ : FaithfulSMul M X
    P Q : M
    h₁ : IsLprojection X P
    h₂ : IsLprojection X Q
    ⊢ ∀ (x : X), Eq (Norm.norm x) (HAdd.hAdd (Norm.norm (HSMul.hSMul (HMul.hMul P  …
  -/
  intro x
  /-
    X : Type u_1
    inst✝³ : NormedAddCommGroup X
    M : Type u_2
    inst✝² : Ring M
    inst✝¹ : Module M X
    inst✝ : FaithfulSMul M X
    P Q : M
    h₁ : IsLprojection X P
    h₂ : IsLprojection X Q
    x : X
    ⊢ Eq (Norm.norm x) (HAdd.hAdd (Norm.norm (HSMul.hSMul (HMul.hMul P Q) x)) (Nor …
  -/
  refine le_antisymm ?_ ?_
  · calc
      ‖x‖ = ‖(P * Q) • x + (x - (P * Q) • x)‖ := by rw [add_sub_cancel ((P * Q) • x) x]
      _ ≤ ‖(P * Q) • x‖ + ‖x - (P * Q) • x‖ := by apply norm_add_le
      _ = ‖(P * Q) • x‖ + ‖(1 - P * Q) • x‖ := by rw [sub_smul, one_smul]
  · calc
      ‖x‖ = ‖P • Q • x‖ + (‖Q • x - P • Q • x‖ + ‖x - Q • x‖) := by
        rw [h₂.Lnorm x, h₁.Lnorm (Q • x), sub_smul, one_smul, sub_smul, one_smul, add_assoc]
      _ ≥ ‖P • Q • x‖ + ‖Q • x - P • Q • x + (x - Q • x)‖ :=
        ((add_le_add_iff_left ‖P • Q • x‖).mpr (norm_add_le (Q • x - P • Q • x) (x - Q • x)))
      _ = ‖(P * Q) • x‖ + ‖(1 - P * Q) • x‖ := by
        rw [sub_add_sub_cancel', sub_smul, one_smul, mul_smul]


theorem join [FaithfulSMul M X] {P Q : M} (h₁ : IsLprojection X P) (h₂ : IsLprojection X Q) :
    IsLprojection X (P + Q - P * Q) := by
  /-
    X : Type u_1
    inst✝³ : NormedAddCommGroup X
    M : Type u_2
    inst✝² : Ring M
    inst✝¹ : Module M X
    inst✝ : FaithfulSMul M X
    P Q : M
    h₁ : IsLprojection X P
    h₂ : IsLprojection X Q
    ⊢ IsLprojection X (HSub.hSub (HAdd.hAdd P Q) (HMul.hMul P Q))
  -/
  convert (Lcomplement_iff _).mp (h₁.Lcomplement.mul h₂.Lcomplement) using 1
  /-
    case h.e'_6
    X : Type u_1
    inst✝³ : NormedAddCommGroup X
    M : Type u_2
    inst✝² : Ring M
    inst✝¹ : Module M X
    inst✝ : FaithfulSMul M X
    P Q : M
    h₁ : IsLprojection X P
    h₂ : IsLprojection X Q
    ⊢ Eq (HSub.hSub (HAdd.hAdd P Q) (HMul.hMul P Q)) (HSub.hSub 1 (HMul.hMul (HSub …
  -/
  noncomm_ring
  /-
    🎉 no goals
  -/

-- Porting note: Advice is to explicitly name instances
-- https://github.com/leanprover-community/mathlib4/wiki/Porting-wiki#some-common-fixes

instance Subtype.hasCompl : HasCompl { f : M // IsLprojection X f } :=
  ⟨fun P => ⟨1 - P, P.prop.Lcomplement⟩⟩


@[simp]
theorem coe_compl (P : { P : M // IsLprojection X P }) : ↑Pᶜ = (1 : M) - ↑P :=
  rfl


instance Subtype.inf [FaithfulSMul M X] : Min { P : M // IsLprojection X P } :=
  ⟨fun P Q => ⟨P * Q, P.prop.mul Q.prop⟩⟩


@[simp]
theorem coe_inf [FaithfulSMul M X] (P Q : { P : M // IsLprojection X P }) :
    ↑(P ⊓ Q) = (↑P : M) * ↑Q :=
  rfl


instance Subtype.sup [FaithfulSMul M X] : Max { P : M // IsLprojection X P } :=
  ⟨fun P Q => ⟨P + Q - P * Q, P.prop.join Q.prop⟩⟩


@[simp]
theorem coe_sup [FaithfulSMul M X] (P Q : { P : M // IsLprojection X P }) :
    ↑(P ⊔ Q) = (↑P : M) + ↑Q - ↑P * ↑Q :=
  rfl


instance Subtype.sdiff [FaithfulSMul M X] : SDiff { P : M // IsLprojection X P } :=
  ⟨fun P Q => ⟨P * (1 - Q), P.prop.mul Q.prop.Lcomplement⟩⟩


@[simp]
theorem coe_sdiff [FaithfulSMul M X] (P Q : { P : M // IsLprojection X P }) :
    ↑(P \ Q) = (↑P : M) * (1 - ↑Q) :=
  rfl


instance Subtype.partialOrder [FaithfulSMul M X] :
    PartialOrder { P : M // IsLprojection X P } where
  le P Q := (↑P : M) = ↑(P ⊓ Q)
                  /-
                    X : Type u_1
                    inst✝³ : NormedAddCommGroup X
                    M : Type u_2
                    inst✝² : Ring M
                    inst✝¹ : Module M X
                    inst✝ : FaithfulSMul M X
                    P : Subtype fun P => IsLprojection X P
                    ⊢ LE.le P P
                  -/
  le_refl P := by simpa only [coe_inf, ← sq] using P.prop.proj.eq.symm
                  /-
                    🎉 no goals
                  -/
  le_trans P Q R h₁ h₂ := by
    /-
      X : Type u_1
      inst✝³ : NormedAddCommGroup X
      M : Type u_2
      inst✝² : Ring M
      inst✝¹ : Module M X
      inst✝ : FaithfulSMul M X
      P Q R : Subtype fun P => IsLprojection X P
      h₁ : LE.le P Q
      h₂ : LE.le Q R
      ⊢ LE.le P R
    -/
    simp only [coe_inf] at h₁ h₂ ⊢
    /-
      X : Type u_1
      inst✝³ : NormedAddCommGroup X
      M : Type u_2
      inst✝² : Ring M
      inst✝¹ : Module M X
      inst✝ : FaithfulSMul M X
      P Q R : Subtype fun P => IsLprojection X P
      h₁ : Eq (↑P) (HMul.hMul ↑P ↑Q)
      h₂ : Eq (↑Q) (HMul.hMul ↑Q ↑R)
      ⊢ Eq (↑P) (HMul.hMul ↑P ↑R)
    -/
    rw [h₁, mul_assoc, ← h₂]
    /-
      🎉 no goals
    -/
                                          /-
                                            X : Type u_1
                                            inst✝³ : NormedAddCommGroup X
                                            M : Type u_2
                                            inst✝² : Ring M
                                            inst✝¹ : Module M X
                                            inst✝ : FaithfulSMul M X
                                            P Q : Subtype fun P => IsLprojection X P
                                            h₁ : LE.le P Q
                                            h₂ : LE.le Q P
                                            ⊢ Eq ↑P ↑Q
                                          -/
  le_antisymm P Q h₁ h₂ := Subtype.eq (by convert (P.prop.commute Q.prop).eq)
                                          /-
                                            🎉 no goals
                                          -/


theorem le_def [FaithfulSMul M X] (P Q : { P : M // IsLprojection X P }) :
    P ≤ Q ↔ (P : M) = ↑(P ⊓ Q) :=
  Iff.rfl


instance Subtype.zero : Zero { P : M // IsLprojection X P } :=
           /-
             X : Type u_1
             inst✝² : NormedAddCommGroup X
             M : Type u_2
             inst✝¹ : Ring M
             inst✝ : Module M X
             ⊢ IsIdempotentElem 0
           -/
  ⟨⟨0, ⟨by rw [IsIdempotentElem, zero_mul], fun x => by
           /-
             🎉 no goals
           -/
        /-
          X : Type u_1
          inst✝² : NormedAddCommGroup X
          M : Type u_2
          inst✝¹ : Ring M
          inst✝ : Module M X
          x : X
          ⊢ Eq (Norm.norm x) (HAdd.hAdd (Norm.norm (HSMul.hSMul 0 x)) (Norm.norm (HSMul. …
        -/
        simp only [zero_smul, norm_zero, sub_zero, one_smul, zero_add]⟩⟩⟩
        /-
          🎉 no goals
        -/


@[simp]
theorem coe_zero : ↑(0 : { P : M // IsLprojection X P }) = (0 : M) :=
  rfl


instance Subtype.one : One { P : M // IsLprojection X P } :=
  ⟨⟨1, sub_zero (1 : M) ▸ (0 : { P : M // IsLprojection X P }).prop.Lcomplement⟩⟩


@[simp]
theorem coe_one : ↑(1 : { P : M // IsLprojection X P }) = (1 : M) :=
  rfl


instance Subtype.boundedOrder [FaithfulSMul M X] :
    BoundedOrder { P : M // IsLprojection X P } where
  top := 1
  le_top P := (mul_one (P : M)).symm
  bot := 0
  bot_le P := (zero_mul (P : M)).symm


@[simp]
theorem coe_bot [FaithfulSMul M X] :
    -- Porting note: Manual correction of name required here
    ↑(BoundedOrder.toOrderBot.toBot.bot : { P : M // IsLprojection X P }) = (0 : M) :=
  rfl


@[simp]
theorem coe_top [FaithfulSMul M X] :
    -- Porting note: Manual correction of name required here
    ↑(BoundedOrder.toOrderTop.toTop.top : { P : M // IsLprojection X P }) = (1 : M) :=
  rfl


theorem compl_mul {P : { P : M // IsLprojection X P }} {Q : M} : ↑Pᶜ * Q = Q - ↑P * Q := by
  /-
    X : Type u_1
    inst✝² : NormedAddCommGroup X
    M : Type u_2
    inst✝¹ : Ring M
    inst✝ : Module M X
    P : Subtype fun P => IsLprojection X P
    Q : M
    ⊢ Eq (HMul.hMul (↑(HasCompl.compl P)) Q) (HSub.hSub Q (HMul.hMul (↑P) Q))
  -/
  rw [coe_compl, sub_mul, one_mul]
  /-
    🎉 no goals
  -/


theorem mul_compl_self {P : { P : M // IsLprojection X P }} : (↑P : M) * ↑Pᶜ = 0 := by
  /-
    X : Type u_1
    inst✝² : NormedAddCommGroup X
    M : Type u_2
    inst✝¹ : Ring M
    inst✝ : Module M X
    P : Subtype fun P => IsLprojection X P
    ⊢ Eq (HMul.hMul ↑P ↑(HasCompl.compl P)) 0
  -/
  rw [coe_compl, mul_sub, mul_one, P.prop.proj.eq, sub_self]
  /-
    🎉 no goals
  -/


theorem distrib_lattice_lemma [FaithfulSMul M X] {P Q R : { P : M // IsLprojection X P }} :
    ((↑P : M) + ↑Pᶜ * R) * (↑P + ↑Q * ↑R * ↑Pᶜ) = ↑P + ↑Q * ↑R * ↑Pᶜ := by
  rw [add_mul, mul_add, mul_add, (mul_assoc _ (R : M) (↑Q * ↑R * ↑Pᶜ)),
    ← mul_assoc (R : M) (↑Q * ↑R) _, ← coe_inf Q, (Pᶜ.prop.commute R.prop).eq,
    ((Q ⊓ R).prop.commute Pᶜ.prop).eq, (R.prop.commute (Q ⊓ R).prop).eq, coe_inf Q,
    mul_assoc (Q : M), ← mul_assoc, mul_assoc (R : M), (Pᶜ.prop.commute P.prop).eq, mul_compl_self,
    zero_mul, mul_zero, zero_add, add_zero, ← mul_assoc, P.prop.proj.eq,
    R.prop.proj.eq, ← coe_inf Q, mul_assoc, ((Q ⊓ R).prop.commute Pᶜ.prop).eq, ← mul_assoc,
    Pᶜ.prop.proj.eq]

-- Porting note: In mathlib3 we were able to directly show that `{ P : M // IsLprojection X P }` was
--  an instance of a `DistribLattice`. Trying to do that in mathlib4 fails with "error:
-- (deterministic) timeout at 'whnf', maximum number of heartbeats (800000) has been reached"
-- My workaround is to show instance Lattice first

instance [FaithfulSMul M X] : Lattice { P : M // IsLprojection X P } where
  sup := max
  inf := min
  le_sup_left P Q := by
    rw [le_def, coe_inf, coe_sup, ← add_sub, mul_add, mul_sub, ← mul_assoc, P.prop.proj.eq,
      sub_self, add_zero]
  le_sup_right P Q := by
    rw [le_def, coe_inf, coe_sup, ← add_sub, mul_add, mul_sub, (P.prop.commute Q.prop).eq,
      ← mul_assoc, Q.prop.proj.eq, add_sub_cancel]
  sup_le P Q R := by
    rw [le_def, le_def, le_def, coe_inf, coe_inf, coe_sup, coe_inf, coe_sup, ← add_sub, add_mul,
      sub_mul, mul_assoc]
    /-
      X : Type u_1
      inst✝³ : NormedAddCommGroup X
      M : Type u_2
      inst✝² : Ring M
      inst✝¹ : Module M X
      inst✝ : FaithfulSMul M X
      P Q R : Subtype fun P => IsLprojection X P
      ⊢ Eq (↑P) (HMul.hMul ↑P ↑R) → Eq (↑Q) (HMul.hMul ↑Q ↑R) → Eq (HAdd.hAdd (↑P) ( …
    -/
    intro h₁ h₂
    /-
      X : Type u_1
      inst✝³ : NormedAddCommGroup X
      M : Type u_2
      inst✝² : Ring M
      inst✝¹ : Module M X
      inst✝ : FaithfulSMul M X
      P Q R : Subtype fun P => IsLprojection X P
      h₁ : Eq (↑P) (HMul.hMul ↑P ↑R)
      h₂ : Eq (↑Q) (HMul.hMul ↑Q ↑R)
      ⊢ Eq (HAdd.hAdd (↑P) (HSub.hSub (↑Q) (HMul.hMul ↑P ↑Q))) (HAdd.hAdd (HMul.hMul …
    -/
    rw [← h₂, ← h₁]
    /-
      🎉 no goals
    -/
  inf_le_left P Q := by
    rw [le_def, coe_inf, coe_inf, coe_inf, mul_assoc, (Q.prop.commute P.prop).eq, ← mul_assoc,
      P.prop.proj.eq]
                         /-
                           X : Type u_1
                           inst✝³ : NormedAddCommGroup X
                           M : Type u_2
                           inst✝² : Ring M
                           inst✝¹ : Module M X
                           inst✝ : FaithfulSMul M X
                           P Q : Subtype fun P => IsLprojection X P
                           ⊢ LE.le (Min.min P Q) Q
                         -/
  inf_le_right P Q := by rw [le_def, coe_inf, coe_inf, coe_inf, mul_assoc, Q.prop.proj.eq]
                         /-
                           🎉 no goals
                         -/
  le_inf P Q R := by
    /-
      X : Type u_1
      inst✝³ : NormedAddCommGroup X
      M : Type u_2
      inst✝² : Ring M
      inst✝¹ : Module M X
      inst✝ : FaithfulSMul M X
      P Q R : Subtype fun P => IsLprojection X P
      ⊢ LE.le P Q → LE.le P R → LE.le P (Min.min Q R)
    -/
    rw [le_def, le_def, le_def, coe_inf, coe_inf, coe_inf, coe_inf, ← mul_assoc]
    /-
      X : Type u_1
      inst✝³ : NormedAddCommGroup X
      M : Type u_2
      inst✝² : Ring M
      inst✝¹ : Module M X
      inst✝ : FaithfulSMul M X
      P Q R : Subtype fun P => IsLprojection X P
      ⊢ Eq (↑P) (HMul.hMul ↑P ↑Q) → Eq (↑P) (HMul.hMul ↑P ↑R) → Eq (↑P) (HMul.hMul ( …
    -/
    intro h₁ h₂
    /-
      X : Type u_1
      inst✝³ : NormedAddCommGroup X
      M : Type u_2
      inst✝² : Ring M
      inst✝¹ : Module M X
      inst✝ : FaithfulSMul M X
      P Q R : Subtype fun P => IsLprojection X P
      h₁ : Eq (↑P) (HMul.hMul ↑P ↑Q)
      h₂ : Eq (↑P) (HMul.hMul ↑P ↑R)
      ⊢ Eq (↑P) (HMul.hMul (HMul.hMul ↑P ↑Q) ↑R)
    -/
    rw [← h₁, ← h₂]
    /-
      🎉 no goals
    -/


instance Subtype.distribLattice [FaithfulSMul M X] :
    DistribLattice { P : M // IsLprojection X P } where
  le_sup_inf P Q R := by
    have e₁ : ↑((P ⊔ Q) ⊓ (P ⊔ R)) = ↑P + ↑Q * (R : M) * ↑Pᶜ := by
      rw [coe_inf, coe_sup, coe_sup, ← add_sub, ← add_sub, ← compl_mul, ← compl_mul, add_mul,
        mul_add, (Pᶜ.prop.commute Q.prop).eq, mul_add, ← mul_assoc, mul_assoc (Q : M),
        (Pᶜ.prop.commute P.prop).eq, mul_compl_self, zero_mul, mul_zero,
        zero_add, add_zero, ← mul_assoc, mul_assoc (Q : M), P.prop.proj.eq, Pᶜ.prop.proj.eq,
        mul_assoc, (Pᶜ.prop.commute R.prop).eq, ← mul_assoc]
    have e₂ : ↑((P ⊔ Q) ⊓ (P ⊔ R)) * ↑(P ⊔ Q ⊓ R) = (P : M) + ↑Q * ↑R * ↑Pᶜ := by
      rw [coe_inf, coe_sup, coe_sup, coe_sup, ← add_sub, ← add_sub, ← add_sub, ← compl_mul, ←
        compl_mul, ← compl_mul, (Pᶜ.prop.commute (Q ⊓ R).prop).eq, coe_inf, mul_assoc,
        distrib_lattice_lemma, (Q.prop.commute R.prop).eq, distrib_lattice_lemma]
    /-
      X : Type u_1
      inst✝³ : NormedAddCommGroup X
      M : Type u_2
      inst✝² : Ring M
      inst✝¹ : Module M X
      inst✝ : FaithfulSMul M X
      P Q R : Subtype fun P => IsLprojection X P
      e₁ : Eq (↑(Min.min (Max.max P Q) (Max.max P R))) (HAdd.hAdd (↑P) (HMul.hMul (H …
      e₂ : Eq (HMul.hMul ↑(Min.min (Max.max P Q) (Max.max P R)) ↑(Max.max P (Min.min …
      ⊢ LE.le (Min.min (Max.max P Q) (Max.max P R)) (Max.max P (Min.min Q R))
    -/
    rw [le_def, e₁, coe_inf, e₂]
    /-
      🎉 no goals
    -/


instance Subtype.BooleanAlgebra [FaithfulSMul M X] :
    BooleanAlgebra { P : M // IsLprojection X P } :=
-- Porting note: use explicitly specified instance names
  { IsLprojection.Subtype.hasCompl,
    IsLprojection.Subtype.sdiff,
    IsLprojection.Subtype.boundedOrder with
    inf_compl_le_bot := fun P =>
                       /-
                         X : Type u_1
                         inst✝³ : NormedAddCommGroup X
                         M : Type u_2
                         inst✝² : Ring M
                         inst✝¹ : Module M X
                         inst✝ : FaithfulSMul M X
                         P : Subtype fun P => IsLprojection X P
                         ⊢ Eq ↑(Min.min P (HasCompl.compl P)) ↑Bot.bot
                       -/
      (Subtype.ext (by rw [coe_inf, coe_compl, coe_bot, ← coe_compl, mul_compl_self])).le
                       /-
                         🎉 no goals
                       -/
    top_le_sup_compl := fun P =>
      (Subtype.ext
        (by
          rw [coe_top, coe_sup, coe_compl, add_sub_cancel, ← coe_compl, mul_compl_self,
            sub_zero])).le
                                             /-
                                               X : Type u_1
                                               inst✝³ : NormedAddCommGroup X
                                               M : Type u_2
                                               inst✝² : Ring M
                                               inst✝¹ : Module M X
                                               inst✝ : FaithfulSMul M X
                                               P Q : Subtype fun P => IsLprojection X P
                                               ⊢ Eq ↑(SDiff.sdiff P Q) ↑(Min.min P (HasCompl.compl Q))
                                             -/
    sdiff_eq := fun P Q => Subtype.ext <| by rw [coe_sdiff, ← coe_compl, coe_inf] }
                                             /-
                                               🎉 no goals
                                             -/


