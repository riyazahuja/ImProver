/-- The Cauchy completion of a ring with absolute value. -/
def Cauchy :=
  @Quotient (CauSeq _ abv) CauSeq.equiv


/-- The map from Cauchy sequences into the Cauchy completion. -/
def mk : CauSeq _ abv → Cauchy abv :=
  Quotient.mk''


@[simp]
theorem mk_eq_mk (f : CauSeq _ abv) : @Eq (Cauchy abv) ⟦f⟧ (mk f) :=
  rfl


theorem mk_eq {f g : CauSeq _ abv} : mk f = mk g ↔ f ≈ g :=
  Quotient.eq


/-- The map from the original ring into the Cauchy completion. -/
def ofRat (x : β) : Cauchy abv :=
  mk (const abv x)


instance : Zero (Cauchy abv) :=
  ⟨ofRat 0⟩


instance : One (Cauchy abv) :=
  ⟨ofRat 1⟩


instance : Inhabited (Cauchy abv) :=
  ⟨0⟩


theorem ofRat_zero : (ofRat 0 : Cauchy abv) = 0 :=
  rfl


theorem ofRat_one : (ofRat 1 : Cauchy abv) = 1 :=
  rfl


@[simp]
theorem mk_eq_zero {f : CauSeq _ abv} : mk f = 0 ↔ LimZero f := by
  /-
    α : Type u_1
    inst✝² : LinearOrderedField α
    β : Type u_2
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : CauSeq β abv
    ⊢ Iff (Eq (CauSeq.Completion.mk f) 0) f.LimZero
  -/
  have : mk f = 0 ↔ LimZero (f - 0) := Quotient.eq
  /-
    α : Type u_1
    inst✝² : LinearOrderedField α
    β : Type u_2
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : CauSeq β abv
    this : Iff (Eq (CauSeq.Completion.mk f) 0) (HSub.hSub f 0).LimZero
    ⊢ Iff (Eq (CauSeq.Completion.mk f) 0) f.LimZero
  -/
  rwa [sub_zero] at this
  /-
    🎉 no goals
  -/


instance : Add (Cauchy abv) :=
  ⟨(Quotient.map₂ (· + ·)) fun _ _ hf _ _ hg => add_equiv_add hf hg⟩


@[simp]
theorem mk_add (f g : CauSeq β abv) : mk f + mk g = mk (f + g) :=
  rfl


instance : Neg (Cauchy abv) :=
  ⟨(Quotient.map Neg.neg) fun _ _ hf => neg_equiv_neg hf⟩


@[simp]
theorem mk_neg (f : CauSeq β abv) : -mk f = mk (-f) :=
  rfl


instance : Mul (Cauchy abv) :=
  ⟨(Quotient.map₂ (· * ·)) fun _ _ hf _ _ hg => mul_equiv_mul hf hg⟩


@[simp]
theorem mk_mul (f g : CauSeq β abv) : mk f * mk g = mk (f * g) :=
  rfl


instance : Sub (Cauchy abv) :=
  ⟨(Quotient.map₂ Sub.sub) fun _ _ hf _ _ hg => sub_equiv_sub hf hg⟩


@[simp]
theorem mk_sub (f g : CauSeq β abv) : mk f - mk g = mk (f - g) :=
  rfl


instance {γ : Type*} [SMul γ β] [IsScalarTower γ β β] : SMul γ (Cauchy abv) :=
  ⟨fun c => (Quotient.map (c • ·)) fun _ _ hf => smul_equiv_smul _ hf⟩


@[simp]
theorem mk_smul {γ : Type*} [SMul γ β] [IsScalarTower γ β β] (c : γ) (f : CauSeq β abv) :
    c • mk f = mk (c • f) :=
  rfl


instance : Pow (Cauchy abv) ℕ :=
  ⟨fun x n => Quotient.map (· ^ n) (fun _ _ hf => pow_equiv_pow hf _) x⟩


@[simp]
theorem mk_pow (n : ℕ) (f : CauSeq β abv) : mk f ^ n = mk (f ^ n) :=
  rfl


instance : NatCast (Cauchy abv) :=
  ⟨fun n => mk n⟩


instance : IntCast (Cauchy abv) :=
  ⟨fun n => mk n⟩


@[simp]
theorem ofRat_natCast (n : ℕ) : (ofRat n : Cauchy abv) = n :=
  rfl


@[simp]
theorem ofRat_intCast (z : ℤ) : (ofRat z : Cauchy abv) = z :=
  rfl


theorem ofRat_add (x y : β) :
    ofRat (x + y) = (ofRat x + ofRat y : Cauchy abv) :=
  congr_arg mk (const_add _ _)


theorem ofRat_neg (x : β) : ofRat (-x) = (-ofRat x : Cauchy abv) :=
  congr_arg mk (const_neg _)


theorem ofRat_mul (x y : β) :
    ofRat (x * y) = (ofRat x * ofRat y : Cauchy abv) :=
  congr_arg mk (const_mul _ _)


private theorem zero_def : 0 = mk (abv := abv) 0 :=
  rfl


private theorem one_def : 1 = mk (abv := abv) 1 :=
  rfl


instance Cauchy.ring : Ring (Cauchy abv) :=
  Function.Surjective.ring mk Quotient.mk'_surjective zero_def.symm one_def.symm
    (fun _ _ => (mk_add _ _).symm) (fun _ _ => (mk_mul _ _).symm) (fun _ => (mk_neg _).symm)
    (fun _ _ => (mk_sub _ _).symm) (fun _ _ => (mk_smul _ _).symm) (fun _ _ => (mk_smul _ _).symm)
    (fun _ _ => (mk_pow _ _).symm) (fun _ => rfl) fun _ => rfl


/-- `CauSeq.Completion.ofRat` as a `RingHom` -/
@[simps]
def ofRatRingHom : β →+* (Cauchy abv) where
  toFun := ofRat
  map_zero' := ofRat_zero
  map_one' := ofRat_one
  map_add' := ofRat_add
  map_mul' := ofRat_mul


theorem ofRat_sub (x y : β) : ofRat (x - y) = (ofRat x - ofRat y : Cauchy abv) :=
  congr_arg mk (const_sub _ _)


instance Cauchy.commRing : CommRing (Cauchy abv) :=
  Function.Surjective.commRing mk Quotient.mk'_surjective zero_def.symm one_def.symm
    (fun _ _ => (mk_add _ _).symm) (fun _ _ => (mk_mul _ _).symm) (fun _ => (mk_neg _).symm)
    (fun _ _ => (mk_sub _ _).symm) (fun _ _ => (mk_smul _ _).symm) (fun _ _ => (mk_smul _ _).symm)
    (fun _ _ => (mk_pow _ _).symm) (fun _ => rfl) fun _ => rfl


instance instNNRatCast : NNRatCast (Cauchy abv) where nnratCast q := ofRat q

instance instRatCast : RatCast (Cauchy abv) where ratCast q := ofRat q


@[simp, norm_cast] lemma ofRat_nnratCast (q : ℚ≥0) : ofRat (q : β) = (q : Cauchy abv) := rfl

@[simp, norm_cast] lemma ofRat_ratCast (q : ℚ) : ofRat (q : β) = (q : Cauchy abv) := rfl


open Classical in
noncomputable instance : Inv (Cauchy abv) :=
  ⟨fun x =>
    (Quotient.liftOn x fun f => mk <| if h : LimZero f then 0 else inv f h) fun f g fg => by
      /-
        α : Type u_1
        inst✝² : LinearOrderedField α
        β : Type u_2
        inst✝¹ : DivisionRing β
        abv : β → α
        inst✝ : IsAbsoluteValue abv
        x : CauSeq.Completion.Cauchy abv
        f g : CauSeq β abv
        fg : HasEquiv.Equiv f g
        ⊢ Eq (CauSeq.Completion.mk (dite f.LimZero (fun h => 0) fun h => f.inv h)) (Ca …
      -/
      have := limZero_congr fg
      /-
        α : Type u_1
        inst✝² : LinearOrderedField α
        β : Type u_2
        inst✝¹ : DivisionRing β
        abv : β → α
        inst✝ : IsAbsoluteValue abv
        x : CauSeq.Completion.Cauchy abv
        f g : CauSeq β abv
        fg : HasEquiv.Equiv f g
        this : Iff f.LimZero g.LimZero
        ⊢ Eq (CauSeq.Completion.mk (dite f.LimZero (fun h => 0) fun h => f.inv h)) (Ca …
      -/
      by_cases hf : LimZero f
        /-
          case pos
          α : Type u_1
          inst✝² : LinearOrderedField α
          β : Type u_2
          inst✝¹ : DivisionRing β
          abv : β → α
          inst✝ : IsAbsoluteValue abv
          x : CauSeq.Completion.Cauchy abv
          f g : CauSeq β abv
          fg : HasEquiv.Equiv f g
          this : Iff f.LimZero g.LimZero
          hf : f.LimZero
          ⊢ Eq (CauSeq.Completion.mk (dite f.LimZero (fun h => 0) fun h => f.inv h)) (Ca …
        -/
      · simp [hf, this.1 hf, Setoid.refl]
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_1
          inst✝² : LinearOrderedField α
          β : Type u_2
          inst✝¹ : DivisionRing β
          abv : β → α
          inst✝ : IsAbsoluteValue abv
          x : CauSeq.Completion.Cauchy abv
          f g : CauSeq β abv
          fg : HasEquiv.Equiv f g
          this : Iff f.LimZero g.LimZero
          hf : Not f.LimZero
          ⊢ Eq (CauSeq.Completion.mk (dite f.LimZero (fun h => 0) fun h => f.inv h)) (Ca …
        -/
      · have hg := mt this.2 hf
        /-
          case neg
          α : Type u_1
          inst✝² : LinearOrderedField α
          β : Type u_2
          inst✝¹ : DivisionRing β
          abv : β → α
          inst✝ : IsAbsoluteValue abv
          x : CauSeq.Completion.Cauchy abv
          f g : CauSeq β abv
          fg : HasEquiv.Equiv f g
          this : Iff f.LimZero g.LimZero
          hf : Not f.LimZero
          hg : Not g.LimZero
          ⊢ Eq (CauSeq.Completion.mk (dite f.LimZero (fun h => 0) fun h => f.inv h)) (Ca …
        -/
        simp only [hf, dite_false, hg]
        /-
          case neg
          α : Type u_1
          inst✝² : LinearOrderedField α
          β : Type u_2
          inst✝¹ : DivisionRing β
          abv : β → α
          inst✝ : IsAbsoluteValue abv
          x : CauSeq.Completion.Cauchy abv
          f g : CauSeq β abv
          fg : HasEquiv.Equiv f g
          this : Iff f.LimZero g.LimZero
          hf : Not f.LimZero
          hg : Not g.LimZero
          ⊢ Eq (CauSeq.Completion.mk (f.inv ⋯)) (CauSeq.Completion.mk (g.inv ⋯))
        -/
        have If : mk (inv f hf) * mk f = 1 := mk_eq.2 (inv_mul_cancel hf)
        /-
          case neg
          α : Type u_1
          inst✝² : LinearOrderedField α
          β : Type u_2
          inst✝¹ : DivisionRing β
          abv : β → α
          inst✝ : IsAbsoluteValue abv
          x : CauSeq.Completion.Cauchy abv
          f g : CauSeq β abv
          fg : HasEquiv.Equiv f g
          this : Iff f.LimZero g.LimZero
          hf : Not f.LimZero
          hg : Not g.LimZero
          If : Eq (HMul.hMul (CauSeq.Completion.mk (f.inv hf)) (CauSeq.Completion.mk f)) 1
          ⊢ Eq (CauSeq.Completion.mk (f.inv ⋯)) (CauSeq.Completion.mk (g.inv ⋯))
        -/
        have Ig : mk (inv g hg) * mk g = 1 := mk_eq.2 (inv_mul_cancel hg)
        /-
          case neg
          α : Type u_1
          inst✝² : LinearOrderedField α
          β : Type u_2
          inst✝¹ : DivisionRing β
          abv : β → α
          inst✝ : IsAbsoluteValue abv
          x : CauSeq.Completion.Cauchy abv
          f g : CauSeq β abv
          fg : HasEquiv.Equiv f g
          this : Iff f.LimZero g.LimZero
          hf : Not f.LimZero
          hg : Not g.LimZero
          If : Eq (HMul.hMul (CauSeq.Completion.mk (f.inv hf)) (CauSeq.Completion.mk f)) 1
          Ig : Eq (HMul.hMul (CauSeq.Completion.mk (g.inv hg)) (CauSeq.Completion.mk g)) 1
          ⊢ Eq (CauSeq.Completion.mk (f.inv ⋯)) (CauSeq.Completion.mk (g.inv ⋯))
        -/
        have Ig' : mk g * mk (inv g hg) = 1 := mk_eq.2 (mul_inv_cancel hg)
        /-
          case neg
          α : Type u_1
          inst✝² : LinearOrderedField α
          β : Type u_2
          inst✝¹ : DivisionRing β
          abv : β → α
          inst✝ : IsAbsoluteValue abv
          x : CauSeq.Completion.Cauchy abv
          f g : CauSeq β abv
          fg : HasEquiv.Equiv f g
          this : Iff f.LimZero g.LimZero
          hf : Not f.LimZero
          hg : Not g.LimZero
          If : Eq (HMul.hMul (CauSeq.Completion.mk (f.inv hf)) (CauSeq.Completion.mk f)) 1
          Ig : Eq (HMul.hMul (CauSeq.Completion.mk (g.inv hg)) (CauSeq.Completion.mk g)) 1
          Ig' : Eq (HMul.hMul (CauSeq.Completion.mk g) (CauSeq.Completion.mk (g.inv hg)) …
          ⊢ Eq (CauSeq.Completion.mk (f.inv ⋯)) (CauSeq.Completion.mk (g.inv ⋯))
        -/
        rw [mk_eq.2 fg, ← Ig] at If
        /-
          case neg
          α : Type u_1
          inst✝² : LinearOrderedField α
          β : Type u_2
          inst✝¹ : DivisionRing β
          abv : β → α
          inst✝ : IsAbsoluteValue abv
          x : CauSeq.Completion.Cauchy abv
          f g : CauSeq β abv
          fg : HasEquiv.Equiv f g
          this : Iff f.LimZero g.LimZero
          hf : Not f.LimZero
          hg : Not g.LimZero
          If : Eq (HMul.hMul (CauSeq.Completion.mk (f.inv hf)) (CauSeq.Completion.mk g)) …
          Ig : Eq (HMul.hMul (CauSeq.Completion.mk (g.inv hg)) (CauSeq.Completion.mk g)) 1
          Ig' : Eq (HMul.hMul (CauSeq.Completion.mk g) (CauSeq.Completion.mk (g.inv hg)) …
          ⊢ Eq (CauSeq.Completion.mk (f.inv ⋯)) (CauSeq.Completion.mk (g.inv ⋯))
        -/
        rw [← mul_one (mk (inv f hf)), ← Ig', ← mul_assoc, If, mul_assoc, Ig', mul_one]⟩
        /-
          🎉 no goals
        -/


theorem inv_zero : (0 : (Cauchy abv))⁻¹ = 0 :=
                     /-
                       α : Type u_1
                       inst✝² : LinearOrderedField α
                       β : Type u_2
                       inst✝¹ : DivisionRing β
                       abv : β → α
                       inst✝ : IsAbsoluteValue abv
                       ⊢ Eq (dite (CauSeq.const abv 0).LimZero (fun h => 0) fun h => (CauSeq.const ab …
                     -/
  congr_arg mk <| by rw [dif_pos] <;> [rfl; exact zero_limZero]
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem inv_mk {f} (hf) : (mk (abv := abv) f)⁻¹ = mk (inv f hf) :=
                     /-
                       α : Type u_1
                       inst✝² : LinearOrderedField α
                       β : Type u_2
                       inst✝¹ : DivisionRing β
                       abv : β → α
                       inst✝ : IsAbsoluteValue abv
                       f : CauSeq β abv
                       hf : Not f.LimZero
                       ⊢ Eq (dite f.LimZero (fun h => 0) fun h => f.inv h) (f.inv hf)
                     -/
  congr_arg mk <| by rw [dif_neg]
                     /-
                       🎉 no goals
                     -/


theorem cau_seq_zero_ne_one : ¬(0 : CauSeq _ abv) ≈ 1 := fun h =>
  have : LimZero (1 - 0 : CauSeq _ abv) := Setoid.symm h
                                          /-
                                            α : Type u_1
                                            inst✝² : LinearOrderedField α
                                            β : Type u_2
                                            inst✝¹ : DivisionRing β
                                            abv : β → α
                                            inst✝ : IsAbsoluteValue abv
                                            h : HasEquiv.Equiv 0 1
                                            this : (HSub.hSub 1 0).LimZero
                                            ⊢ CauSeq.LimZero 1
                                          -/
  have : LimZero (1 : CauSeq _ abv) := by simpa
                                          /-
                                            🎉 no goals
                                          -/
     /-
       α : Type u_1
       inst✝² : LinearOrderedField α
       β : Type u_2
       inst✝¹ : DivisionRing β
       abv : β → α
       inst✝ : IsAbsoluteValue abv
       h : HasEquiv.Equiv 0 1
       this✝ : (HSub.hSub 1 0).LimZero
       this : CauSeq.LimZero 1
       ⊢ False
     -/
  by apply one_ne_zero <| const_limZero.1 this
     /-
       🎉 no goals
     -/


theorem zero_ne_one : (0 : (Cauchy abv)) ≠ 1 := fun h => cau_seq_zero_ne_one <| mk_eq.1 h


protected theorem inv_mul_cancel {x : (Cauchy abv)} : x ≠ 0 → x⁻¹ * x = 1 :=
  Quotient.inductionOn x fun f hf => by
    /-
      α : Type u_1
      inst✝² : LinearOrderedField α
      β : Type u_2
      inst✝¹ : DivisionRing β
      abv : β → α
      inst✝ : IsAbsoluteValue abv
      x : CauSeq.Completion.Cauchy abv
      f : CauSeq β abv
      hf : Ne (Quotient.mk CauSeq.equiv f) 0
      ⊢ Eq (HMul.hMul (Inv.inv (Quotient.mk CauSeq.equiv f)) (Quotient.mk CauSeq.equ …
    -/
    simp only [mk_eq_mk, ne_eq, mk_eq_zero] at hf
    /-
      α : Type u_1
      inst✝² : LinearOrderedField α
      β : Type u_2
      inst✝¹ : DivisionRing β
      abv : β → α
      inst✝ : IsAbsoluteValue abv
      x : CauSeq.Completion.Cauchy abv
      f : CauSeq β abv
      hf : Not f.LimZero
      ⊢ Eq (HMul.hMul (Inv.inv (Quotient.mk CauSeq.equiv f)) (Quotient.mk CauSeq.equ …
    -/
    simp only [mk_eq_mk, hf, not_false_eq_true, inv_mk, mk_mul]
    /-
      α : Type u_1
      inst✝² : LinearOrderedField α
      β : Type u_2
      inst✝¹ : DivisionRing β
      abv : β → α
      inst✝ : IsAbsoluteValue abv
      x : CauSeq.Completion.Cauchy abv
      f : CauSeq β abv
      hf : Not f.LimZero
      ⊢ Eq (CauSeq.Completion.mk (HMul.hMul (f.inv ⋯) f)) 1
    -/
    exact Quotient.sound (CauSeq.inv_mul_cancel hf)
    /-
      🎉 no goals
    -/


protected theorem mul_inv_cancel {x : (Cauchy abv)} : x ≠ 0 → x * x⁻¹ = 1 :=
  Quotient.inductionOn x fun f hf => by
    /-
      α : Type u_1
      inst✝² : LinearOrderedField α
      β : Type u_2
      inst✝¹ : DivisionRing β
      abv : β → α
      inst✝ : IsAbsoluteValue abv
      x : CauSeq.Completion.Cauchy abv
      f : CauSeq β abv
      hf : Ne (Quotient.mk CauSeq.equiv f) 0
      ⊢ Eq (HMul.hMul (Quotient.mk CauSeq.equiv f) (Inv.inv (Quotient.mk CauSeq.equi …
    -/
    simp only [mk_eq_mk, ne_eq, mk_eq_zero] at hf
    /-
      α : Type u_1
      inst✝² : LinearOrderedField α
      β : Type u_2
      inst✝¹ : DivisionRing β
      abv : β → α
      inst✝ : IsAbsoluteValue abv
      x : CauSeq.Completion.Cauchy abv
      f : CauSeq β abv
      hf : Not f.LimZero
      ⊢ Eq (HMul.hMul (Quotient.mk CauSeq.equiv f) (Inv.inv (Quotient.mk CauSeq.equi …
    -/
    simp only [mk_eq_mk, hf, not_false_eq_true, inv_mk, mk_mul]
    /-
      α : Type u_1
      inst✝² : LinearOrderedField α
      β : Type u_2
      inst✝¹ : DivisionRing β
      abv : β → α
      inst✝ : IsAbsoluteValue abv
      x : CauSeq.Completion.Cauchy abv
      f : CauSeq β abv
      hf : Not f.LimZero
      ⊢ Eq (CauSeq.Completion.mk (HMul.hMul f (f.inv ⋯))) 1
    -/
    exact Quotient.sound (CauSeq.mul_inv_cancel hf)
    /-
      🎉 no goals
    -/


theorem ofRat_inv (x : β) : ofRat x⁻¹ = ((ofRat x)⁻¹ : (Cauchy abv)) :=
  congr_arg mk <| by split_ifs with h <;>
    [simp only [const_limZero.1 h, GroupWithZero.inv_zero, const_zero]; rfl]


noncomputable instance instDivInvMonoid : DivInvMonoid (Cauchy abv) where


lemma ofRat_div (x y : β) : ofRat (x / y) = (ofRat x / ofRat y : Cauchy abv) := by
  /-
    α : Type u_1
    inst✝² : LinearOrderedField α
    β : Type u_2
    inst✝¹ : DivisionRing β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    x y : β
    ⊢ Eq (CauSeq.Completion.ofRat (HDiv.hDiv x y)) (HDiv.hDiv (CauSeq.Completion.o …
  -/
  simp only [div_eq_mul_inv, ofRat_inv, ofRat_mul]
  /-
    🎉 no goals
  -/


/-- The Cauchy completion forms a division ring. -/
noncomputable instance Cauchy.divisionRing : DivisionRing (Cauchy abv) where
  exists_pair_ne := ⟨0, 1, zero_ne_one⟩
  inv_zero := inv_zero
  mul_inv_cancel _ := CauSeq.Completion.mul_inv_cancel
  nnqsmul := (· • ·)
  qsmul := (· • ·)
                        /-
                          α : Type u_1
                          inst✝² : LinearOrderedField α
                          β : Type u_2
                          inst✝¹ : DivisionRing β
                          abv : β → α
                          inst✝ : IsAbsoluteValue abv
                          q : NNRat
                          ⊢ Eq (↑q) (HDiv.hDiv ↑q.num ↑q.den)
                        -/
  nnratCast_def q := by simp_rw [← ofRat_nnratCast, NNRat.cast_def, ofRat_div, ofRat_natCast]
                        /-
                          🎉 no goals
                        -/
                      /-
                        α : Type u_1
                        inst✝² : LinearOrderedField α
                        β : Type u_2
                        inst✝¹ : DivisionRing β
                        abv : β → α
                        inst✝ : IsAbsoluteValue abv
                        q : Rat
                        ⊢ Eq (↑q) (HDiv.hDiv ↑q.num ↑q.den)
                      -/
  ratCast_def q := by rw [← ofRat_ratCast, Rat.cast_def, ofRat_div, ofRat_natCast, ofRat_intCast]
                      /-
                        🎉 no goals
                      -/
  nnqsmul_def _ x := Quotient.inductionOn x fun _ ↦ congr_arg mk <| ext fun _ ↦ NNRat.smul_def _ _
  qsmul_def _ x := Quotient.inductionOn x fun _ ↦ congr_arg mk <| ext fun _ ↦ Rat.smul_def _ _


/-- Show the first 10 items of a representative of this equivalence class of cauchy sequences.

The representative chosen is the one passed in the VM to `Quot.mk`, so two cauchy sequences
converging to the same number may be printed differently.
-/
unsafe instance [Repr β] : Repr (Cauchy abv) where
  reprPrec r _ :=
    let N := 10
    let seq := r.unquot
    "(sorry /- " ++ Std.Format.joinSep ((List.range N).map <| repr ∘ seq) ", " ++ ", ... -/)"


/-- The Cauchy completion forms a field. -/
noncomputable instance Cauchy.field : Field (Cauchy abv) :=
  { Cauchy.divisionRing, Cauchy.commRing with }


/-- A class stating that a ring with an absolute value is complete, i.e. every Cauchy
sequence has a limit. -/
class IsComplete : Prop where
  /-- Every Cauchy sequence has a limit. -/
  isComplete : ∀ s : CauSeq β abv, ∃ b : β, s ≈ const abv b


theorem complete : ∀ s : CauSeq β abv, ∃ b : β, s ≈ const abv b :=
  IsComplete.isComplete


/-- The limit of a Cauchy sequence in a complete ring. Chosen non-computably. -/
noncomputable def lim (s : CauSeq β abv) : β :=
  Classical.choose (complete s)


theorem equiv_lim (s : CauSeq β abv) : s ≈ const abv (lim s) :=
  Classical.choose_spec (complete s)


theorem eq_lim_of_const_equiv {f : CauSeq β abv} {x : β} (h : CauSeq.const abv x ≈ f) : x = lim f :=
  const_equiv.mp <| Setoid.trans h <| equiv_lim f


theorem lim_eq_of_equiv_const {f : CauSeq β abv} {x : β} (h : f ≈ CauSeq.const abv x) : lim f = x :=
  (eq_lim_of_const_equiv <| Setoid.symm h).symm


theorem lim_eq_lim_of_equiv {f g : CauSeq β abv} (h : f ≈ g) : lim f = lim g :=
  lim_eq_of_equiv_const <| Setoid.trans h <| equiv_lim g


@[simp]
theorem lim_const (x : β) : lim (const abv x) = x :=
  lim_eq_of_equiv_const <| Setoid.refl _


theorem lim_add (f g : CauSeq β abv) : lim f + lim g = lim (f + g) :=
  eq_lim_of_const_equiv <|
    show LimZero (const abv (lim f + lim g) - (f + g)) by
      /-
        α : Type u_1
        inst✝³ : LinearOrderedField α
        β : Type u_2
        inst✝² : Ring β
        abv : β → α
        inst✝¹ : IsAbsoluteValue abv
        inst✝ : CauSeq.IsComplete β abv
        f g : CauSeq β abv
        ⊢ (HSub.hSub (CauSeq.const abv (HAdd.hAdd f.lim g.lim)) (HAdd.hAdd f g)).LimZero
      -/
      rw [const_add, add_sub_add_comm]
      /-
        α : Type u_1
        inst✝³ : LinearOrderedField α
        β : Type u_2
        inst✝² : Ring β
        abv : β → α
        inst✝¹ : IsAbsoluteValue abv
        inst✝ : CauSeq.IsComplete β abv
        f g : CauSeq β abv
        ⊢ (HAdd.hAdd (HSub.hSub (CauSeq.const abv f.lim) f) (HSub.hSub (CauSeq.const a …
      -/
      exact add_limZero (Setoid.symm (equiv_lim f)) (Setoid.symm (equiv_lim g))
      /-
        🎉 no goals
      -/


theorem lim_mul_lim (f g : CauSeq β abv) : lim f * lim g = lim (f * g) :=
  eq_lim_of_const_equiv <|
    show LimZero (const abv (lim f * lim g) - f * g) by
      have h :
        const abv (lim f * lim g) - f * g =
          (const abv (lim f) - f) * g + const abv (lim f) * (const abv (lim g) - g) := by
              apply Subtype.ext
              rw [coe_add]
              simp [sub_mul, mul_sub]
      /-
        α : Type u_1
        inst✝³ : LinearOrderedField α
        β : Type u_2
        inst✝² : Ring β
        abv : β → α
        inst✝¹ : IsAbsoluteValue abv
        inst✝ : CauSeq.IsComplete β abv
        f g : CauSeq β abv
        h : Eq (HSub.hSub (CauSeq.const abv (HMul.hMul f.lim g.lim)) (HMul.hMul f g))  …
        ⊢ (HSub.hSub (CauSeq.const abv (HMul.hMul f.lim g.lim)) (HMul.hMul f g)).LimZero
      -/
      rw [h]
      exact
        add_limZero (mul_limZero_left _ (Setoid.symm (equiv_lim _)))
          (mul_limZero_right _ (Setoid.symm (equiv_lim _)))


theorem lim_mul (f : CauSeq β abv) (x : β) : lim f * x = lim (f * const abv x) := by
  /-
    α : Type u_1
    inst✝³ : LinearOrderedField α
    β : Type u_2
    inst✝² : Ring β
    abv : β → α
    inst✝¹ : IsAbsoluteValue abv
    inst✝ : CauSeq.IsComplete β abv
    f : CauSeq β abv
    x : β
    ⊢ Eq (HMul.hMul f.lim x) (HMul.hMul f (CauSeq.const abv x)).lim
  -/
  rw [← lim_mul_lim, lim_const]
  /-
    🎉 no goals
  -/


theorem lim_neg (f : CauSeq β abv) : lim (-f) = -lim f :=
  lim_eq_of_equiv_const
    (show LimZero (-f - const abv (-lim f)) by
      /-
        α : Type u_1
        inst✝³ : LinearOrderedField α
        β : Type u_2
        inst✝² : Ring β
        abv : β → α
        inst✝¹ : IsAbsoluteValue abv
        inst✝ : CauSeq.IsComplete β abv
        f : CauSeq β abv
        ⊢ (HSub.hSub (Neg.neg f) (CauSeq.const abv (Neg.neg f.lim))).LimZero
      -/
      rw [const_neg, sub_neg_eq_add, add_comm, ← sub_eq_add_neg]
      /-
        α : Type u_1
        inst✝³ : LinearOrderedField α
        β : Type u_2
        inst✝² : Ring β
        abv : β → α
        inst✝¹ : IsAbsoluteValue abv
        inst✝ : CauSeq.IsComplete β abv
        f : CauSeq β abv
        ⊢ (HSub.hSub (CauSeq.const abv f.lim) f).LimZero
      -/
      exact Setoid.symm (equiv_lim f))
      /-
        🎉 no goals
      -/


theorem lim_eq_zero_iff (f : CauSeq β abv) : lim f = 0 ↔ LimZero f :=
  ⟨fun h => by
    /-
      α : Type u_1
      inst✝³ : LinearOrderedField α
      β : Type u_2
      inst✝² : Ring β
      abv : β → α
      inst✝¹ : IsAbsoluteValue abv
      inst✝ : CauSeq.IsComplete β abv
      f : CauSeq β abv
      h : Eq f.lim 0
      ⊢ f.LimZero
    -/
    have hf := equiv_lim f
    /-
      α : Type u_1
      inst✝³ : LinearOrderedField α
      β : Type u_2
      inst✝² : Ring β
      abv : β → α
      inst✝¹ : IsAbsoluteValue abv
      inst✝ : CauSeq.IsComplete β abv
      f : CauSeq β abv
      h : Eq f.lim 0
      hf : HasEquiv.Equiv f (CauSeq.const abv f.lim)
      ⊢ f.LimZero
    -/
    rw [h] at hf
    /-
      α : Type u_1
      inst✝³ : LinearOrderedField α
      β : Type u_2
      inst✝² : Ring β
      abv : β → α
      inst✝¹ : IsAbsoluteValue abv
      inst✝ : CauSeq.IsComplete β abv
      f : CauSeq β abv
      h : Eq f.lim 0
      hf : HasEquiv.Equiv f (CauSeq.const abv 0)
      ⊢ f.LimZero
    -/
    exact (limZero_congr hf).mpr (const_limZero.mpr rfl),
    /-
      🎉 no goals
    -/
   fun h => by
    /-
      α : Type u_1
      inst✝³ : LinearOrderedField α
      β : Type u_2
      inst✝² : Ring β
      abv : β → α
      inst✝¹ : IsAbsoluteValue abv
      inst✝ : CauSeq.IsComplete β abv
      f : CauSeq β abv
      h : f.LimZero
      ⊢ Eq f.lim 0
    -/
    have h₁ : f = f - const abv 0 := ext fun n => by simp [sub_apply, const_apply]
    /-
      α : Type u_1
      inst✝³ : LinearOrderedField α
      β : Type u_2
      inst✝² : Ring β
      abv : β → α
      inst✝¹ : IsAbsoluteValue abv
      inst✝ : CauSeq.IsComplete β abv
      f : CauSeq β abv
      h : f.LimZero
      h₁ : Eq f (HSub.hSub f (CauSeq.const abv 0))
      ⊢ Eq f.lim 0
    -/
    rw [h₁] at h
    /-
      α : Type u_1
      inst✝³ : LinearOrderedField α
      β : Type u_2
      inst✝² : Ring β
      abv : β → α
      inst✝¹ : IsAbsoluteValue abv
      inst✝ : CauSeq.IsComplete β abv
      f : CauSeq β abv
      h : (HSub.hSub f (CauSeq.const abv 0)).LimZero
      h₁ : Eq f (HSub.hSub f (CauSeq.const abv 0))
      ⊢ Eq f.lim 0
    -/
    exact lim_eq_of_equiv_const h⟩
    /-
      🎉 no goals
    -/


theorem lim_inv {f : CauSeq β abv} (hf : ¬LimZero f) : lim (inv f hf) = (lim f)⁻¹ :=
                            /-
                              α : Type u_1
                              inst✝³ : LinearOrderedField α
                              β : Type u_2
                              inst✝² : Field β
                              abv : β → α
                              inst✝¹ : IsAbsoluteValue abv
                              inst✝ : CauSeq.IsComplete β abv
                              f : CauSeq β abv
                              hf : Not f.LimZero
                              ⊢ Ne f.lim 0
                            -/
  have hl : lim f ≠ 0 := by rwa [← lim_eq_zero_iff] at hf
                            /-
                              🎉 no goals
                            -/
  lim_eq_of_equiv_const <|
    show LimZero (inv f hf - const abv (lim f)⁻¹) from
      have h₁ : ∀ (g f : CauSeq β abv) (hf : ¬LimZero f), LimZero (g - f * inv f hf * g) :=
        fun g f hf => by
          /-
            α : Type u_1
            inst✝³ : LinearOrderedField α
            β : Type u_2
            inst✝² : Field β
            abv : β → α
            inst✝¹ : IsAbsoluteValue abv
            inst✝ : CauSeq.IsComplete β abv
            f✝ : CauSeq β abv
            hf✝ : Not f✝.LimZero
            hl : Ne f✝.lim 0
            g f : CauSeq β abv
            hf : Not f.LimZero
            ⊢ (HSub.hSub g (HMul.hMul (HMul.hMul f (f.inv hf)) g)).LimZero
          -/
          have h₂ : g - f * inv f hf * g = 1 * g - f * inv f hf * g := by rw [one_mul g]
          /-
            α : Type u_1
            inst✝³ : LinearOrderedField α
            β : Type u_2
            inst✝² : Field β
            abv : β → α
            inst✝¹ : IsAbsoluteValue abv
            inst✝ : CauSeq.IsComplete β abv
            f✝ : CauSeq β abv
            hf✝ : Not f✝.LimZero
            hl : Ne f✝.lim 0
            g f : CauSeq β abv
            hf : Not f.LimZero
            h₂ : Eq (HSub.hSub g (HMul.hMul (HMul.hMul f (f.inv hf)) g)) (HSub.hSub (HMul. …
            ⊢ (HSub.hSub g (HMul.hMul (HMul.hMul f (f.inv hf)) g)).LimZero
          -/
          have h₃ : f * inv f hf * g = (f * inv f hf) * g := by simp [mul_assoc]
          /-
            α : Type u_1
            inst✝³ : LinearOrderedField α
            β : Type u_2
            inst✝² : Field β
            abv : β → α
            inst✝¹ : IsAbsoluteValue abv
            inst✝ : CauSeq.IsComplete β abv
            f✝ : CauSeq β abv
            hf✝ : Not f✝.LimZero
            hl : Ne f✝.lim 0
            g f : CauSeq β abv
            hf : Not f.LimZero
            h₂ : Eq (HSub.hSub g (HMul.hMul (HMul.hMul f (f.inv hf)) g)) (HSub.hSub (HMul. …
            h₃ : Eq (HMul.hMul (HMul.hMul f (f.inv hf)) g) (HMul.hMul (HMul.hMul f (f.inv  …
            ⊢ (HSub.hSub g (HMul.hMul (HMul.hMul f (f.inv hf)) g)).LimZero
          -/
          have h₄ : g - f * inv f hf * g = (1 - f * inv f hf) * g := by rw [h₂, h₃, ← sub_mul]
          /-
            α : Type u_1
            inst✝³ : LinearOrderedField α
            β : Type u_2
            inst✝² : Field β
            abv : β → α
            inst✝¹ : IsAbsoluteValue abv
            inst✝ : CauSeq.IsComplete β abv
            f✝ : CauSeq β abv
            hf✝ : Not f✝.LimZero
            hl : Ne f✝.lim 0
            g f : CauSeq β abv
            hf : Not f.LimZero
            h₂ : Eq (HSub.hSub g (HMul.hMul (HMul.hMul f (f.inv hf)) g)) (HSub.hSub (HMul. …
            h₃ : Eq (HMul.hMul (HMul.hMul f (f.inv hf)) g) (HMul.hMul (HMul.hMul f (f.inv  …
            h₄ : Eq (HSub.hSub g (HMul.hMul (HMul.hMul f (f.inv hf)) g)) (HMul.hMul (HSub. …
            ⊢ (HSub.hSub g (HMul.hMul (HMul.hMul f (f.inv hf)) g)).LimZero
          -/
          have h₅ : g - f * inv f hf * g = g * (1 - f * inv f hf) := by rw [h₄, mul_comm]
          /-
            α : Type u_1
            inst✝³ : LinearOrderedField α
            β : Type u_2
            inst✝² : Field β
            abv : β → α
            inst✝¹ : IsAbsoluteValue abv
            inst✝ : CauSeq.IsComplete β abv
            f✝ : CauSeq β abv
            hf✝ : Not f✝.LimZero
            hl : Ne f✝.lim 0
            g f : CauSeq β abv
            hf : Not f.LimZero
            h₂ : Eq (HSub.hSub g (HMul.hMul (HMul.hMul f (f.inv hf)) g)) (HSub.hSub (HMul. …
            h₃ : Eq (HMul.hMul (HMul.hMul f (f.inv hf)) g) (HMul.hMul (HMul.hMul f (f.inv  …
            h₄ : Eq (HSub.hSub g (HMul.hMul (HMul.hMul f (f.inv hf)) g)) (HMul.hMul (HSub. …
            h₅ : Eq (HSub.hSub g (HMul.hMul (HMul.hMul f (f.inv hf)) g)) (HMul.hMul g (HSu …
            ⊢ (HSub.hSub g (HMul.hMul (HMul.hMul f (f.inv hf)) g)).LimZero
          -/
          have h₆ : g - f * inv f hf * g = g * (1 - inv f hf * f) := by rw [h₅, mul_comm f]
          /-
            α : Type u_1
            inst✝³ : LinearOrderedField α
            β : Type u_2
            inst✝² : Field β
            abv : β → α
            inst✝¹ : IsAbsoluteValue abv
            inst✝ : CauSeq.IsComplete β abv
            f✝ : CauSeq β abv
            hf✝ : Not f✝.LimZero
            hl : Ne f✝.lim 0
            g f : CauSeq β abv
            hf : Not f.LimZero
            h₂ : Eq (HSub.hSub g (HMul.hMul (HMul.hMul f (f.inv hf)) g)) (HSub.hSub (HMul. …
            h₃ : Eq (HMul.hMul (HMul.hMul f (f.inv hf)) g) (HMul.hMul (HMul.hMul f (f.inv  …
            h₄ : Eq (HSub.hSub g (HMul.hMul (HMul.hMul f (f.inv hf)) g)) (HMul.hMul (HSub. …
            h₅ : Eq (HSub.hSub g (HMul.hMul (HMul.hMul f (f.inv hf)) g)) (HMul.hMul g (HSu …
            h₆ : Eq (HSub.hSub g (HMul.hMul (HMul.hMul f (f.inv hf)) g)) (HMul.hMul g (HSu …
            ⊢ (HSub.hSub g (HMul.hMul (HMul.hMul f (f.inv hf)) g)).LimZero
          -/
          rw [h₆]; exact mul_limZero_right _ (Setoid.symm (CauSeq.inv_mul_cancel _))
                   /-
                     🎉 no goals
                   -/
      have h₂ :
        LimZero
          (inv f hf - const abv (lim f)⁻¹ -
            (const abv (lim f) - f) * (inv f hf * const abv (lim f)⁻¹)) := by
              /-
                α : Type u_1
                inst✝³ : LinearOrderedField α
                β : Type u_2
                inst✝² : Field β
                abv : β → α
                inst✝¹ : IsAbsoluteValue abv
                inst✝ : CauSeq.IsComplete β abv
                f : CauSeq β abv
                hf : Not f.LimZero
                hl : Ne f.lim 0
                h₁ : ∀ (g f : CauSeq β abv) (hf : Not f.LimZero), (HSub.hSub g (HMul.hMul (HMu …
                ⊢ (HSub.hSub (HSub.hSub (f.inv hf) (CauSeq.const abv (Inv.inv f.lim))) (HMul.h …
              -/
              rw [sub_mul, ← sub_add, sub_sub, sub_add_eq_sub_sub, sub_right_comm, sub_add]
              show LimZero
                (inv f hf - const abv (lim f) * (inv f hf * const abv (lim f)⁻¹) -
                  (const abv (lim f)⁻¹ - f * (inv f hf * const abv (lim f)⁻¹)))
              exact sub_limZero
                (by rw [← mul_assoc, mul_right_comm, const_inv hl]; exact h₁ _ _ _)
                (by rw [← mul_assoc]; exact h₁ _ _ _)
      (limZero_congr h₂).mpr <| mul_limZero_left _ (Setoid.symm (equiv_lim f))


theorem lim_le {f : CauSeq α abs} {x : α} (h : f ≤ CauSeq.const abs x) : lim f ≤ x :=
  CauSeq.const_le.1 <| CauSeq.le_of_eq_of_le (Setoid.symm (equiv_lim f)) h


theorem le_lim {f : CauSeq α abs} {x : α} (h : CauSeq.const abs x ≤ f) : x ≤ lim f :=
  CauSeq.const_le.1 <| CauSeq.le_of_le_of_eq h (equiv_lim f)


theorem lt_lim {f : CauSeq α abs} {x : α} (h : CauSeq.const abs x < f) : x < lim f :=
  CauSeq.const_lt.1 <| CauSeq.lt_of_lt_of_eq h (equiv_lim f)


theorem lim_lt {f : CauSeq α abs} {x : α} (h : f < CauSeq.const abs x) : lim f < x :=
  CauSeq.const_lt.1 <| CauSeq.lt_of_eq_of_lt (Setoid.symm (equiv_lim f)) h


