instance one : One (Completion α) :=
  ⟨(1 : α)⟩


instance mul : Mul (Completion α) :=
  ⟨curry <| (isDenseInducing_coe.prodMap isDenseInducing_coe).extend ((↑) ∘ uncurry (· * ·))⟩


@[norm_cast]
theorem coe_one : ((1 : α) : Completion α) = 1 :=
  rfl


@[norm_cast]
theorem coe_mul (a b : α) : ((a * b : α) : Completion α) = a * b :=
  ((isDenseInducing_coe.prodMap isDenseInducing_coe).extend_eq
      ((continuous_coe α).comp (@continuous_mul α _ _ _)) (a, b)).symm


instance : ContinuousMul (Completion α) where
  continuous_mul := by
    /-
      α : Type u_1
      inst✝³ : Ring α
      inst✝² : UniformSpace α
      inst✝¹ : TopologicalRing α
      inst✝ : UniformAddGroup α
      ⊢ Continuous fun p => HMul.hMul p.1 p.2
    -/
    let m := (AddMonoidHom.mul : α →+ α →+ α).compr₂ toCompl
    /-
      α : Type u_1
      inst✝³ : Ring α
      inst✝² : UniformSpace α
      inst✝¹ : TopologicalRing α
      inst✝ : UniformAddGroup α
      m : AddMonoidHom α (AddMonoidHom α (UniformSpace.Completion α)) := AddMonoidHo …
      ⊢ Continuous fun p => HMul.hMul p.1 p.2
    -/
    have : Continuous fun p : α × α => m p.1 p.2 := (continuous_coe α).comp continuous_mul
    /-
      α : Type u_1
      inst✝³ : Ring α
      inst✝² : UniformSpace α
      inst✝¹ : TopologicalRing α
      inst✝ : UniformAddGroup α
      m : AddMonoidHom α (AddMonoidHom α (UniformSpace.Completion α)) := AddMonoidHo …
      this : Continuous fun p => (m p.1) p.2
      ⊢ Continuous fun p => HMul.hMul p.1 p.2
    -/
    have di : IsDenseInducing (toCompl : α → Completion α) := isDenseInducing_coe
    /-
      α : Type u_1
      inst✝³ : Ring α
      inst✝² : UniformSpace α
      inst✝¹ : TopologicalRing α
      inst✝ : UniformAddGroup α
      m : AddMonoidHom α (AddMonoidHom α (UniformSpace.Completion α)) := AddMonoidHo …
      this : Continuous fun p => (m p.1) p.2
      di : IsDenseInducing ⇑UniformSpace.Completion.toCompl
      ⊢ Continuous fun p => HMul.hMul p.1 p.2
    -/
    exact (di.extend_Z_bilin di this :)
    /-
      🎉 no goals
    -/


@[deprecated _root_.continuous_mul (since := "2024-12-21")]
protected theorem continuous_mul : Continuous fun p : Completion α × Completion α => p.1 * p.2 :=
  _root_.continuous_mul


@[deprecated _root_.Continuous.mul (since := "2024-12-21")]
protected theorem Continuous.mul {β : Type*} [TopologicalSpace β] {f g : β → Completion α}
    (hf : Continuous f) (hg : Continuous g) : Continuous fun b => f b * g b :=
  hf.mul hg


instance ring : Ring (Completion α) :=
  { AddMonoidWithOne.unary, (inferInstanceAs (AddCommGroup (Completion α))),
      (inferInstanceAs (Mul (Completion α))), (inferInstanceAs (One (Completion α))) with
    zero_mul := fun a =>
      Completion.induction_on a
        (isClosed_eq (continuous_const.mul continuous_id) continuous_const)
                    /-
                      α : Type u_1
                      inst✝³ : Ring α
                      inst✝² : UniformSpace α
                      inst✝¹ : TopologicalRing α
                      inst✝ : UniformAddGroup α
                      a✝ : UniformSpace.Completion α
                      a : α
                      ⊢ Eq (HMul.hMul 0 (↑α a)) 0
                    -/
        fun a => by rw [← coe_zero, ← coe_mul, zero_mul]
                    /-
                      🎉 no goals
                    -/
    mul_zero := fun a =>
      Completion.induction_on a
        (isClosed_eq (continuous_id.mul continuous_const) continuous_const)
                    /-
                      α : Type u_1
                      inst✝³ : Ring α
                      inst✝² : UniformSpace α
                      inst✝¹ : TopologicalRing α
                      inst✝ : UniformAddGroup α
                      a✝ : UniformSpace.Completion α
                      a : α
                      ⊢ Eq (HMul.hMul (↑α a) 0) 0
                    -/
        fun a => by rw [← coe_zero, ← coe_mul, mul_zero]
                    /-
                      🎉 no goals
                    -/
    one_mul := fun a =>
      Completion.induction_on a
        (isClosed_eq (continuous_const.mul continuous_id) continuous_id) fun a => by
        /-
          α : Type u_1
          inst✝³ : Ring α
          inst✝² : UniformSpace α
          inst✝¹ : TopologicalRing α
          inst✝ : UniformAddGroup α
          a✝ : UniformSpace.Completion α
          a : α
          ⊢ Eq (HMul.hMul 1 (↑α a)) (↑α a)
        -/
        rw [← coe_one, ← coe_mul, one_mul]
        /-
          🎉 no goals
        -/
    mul_one := fun a =>
      Completion.induction_on a
        (isClosed_eq (continuous_id.mul continuous_const) continuous_id) fun a => by
        /-
          α : Type u_1
          inst✝³ : Ring α
          inst✝² : UniformSpace α
          inst✝¹ : TopologicalRing α
          inst✝ : UniformAddGroup α
          a✝ : UniformSpace.Completion α
          a : α
          ⊢ Eq (HMul.hMul (↑α a) 1) (↑α a)
        -/
        rw [← coe_one, ← coe_mul, mul_one]
        /-
          🎉 no goals
        -/
    mul_assoc := fun a b c =>
      Completion.induction_on₃ a b c
        (isClosed_eq
          ((continuous_fst.mul (continuous_fst.comp continuous_snd)).mul
                                /-
                                  α : Type u_1
                                  inst✝³ : Ring α
                                  inst✝² : UniformSpace α
                                  inst✝¹ : TopologicalRing α
                                  inst✝ : UniformAddGroup α
                                  a✝ b✝ c✝ : UniformSpace.Completion α
                                  a b c : α
                                  ⊢ Eq (HMul.hMul (HMul.hMul (↑α a) (↑α b)) (↑α c)) (HMul.hMul (↑α a) (HMul.hMul …
                                -/
            (continuous_snd.comp continuous_snd))
                                /-
                                  🎉 no goals
                                -/
          (continuous_fst.mul
            ((continuous_fst.comp continuous_snd).mul
                        /-
                          α : Type u_1
                          inst✝³ : Ring α
                          inst✝² : UniformSpace α
                          inst✝¹ : TopologicalRing α
                          inst✝ : UniformAddGroup α
                          a✝ b✝ c✝ : UniformSpace.Completion α
                          a b c : α
                          ⊢ Eq (HMul.hMul (↑α a) (HAdd.hAdd (↑α b) (↑α c))) (HAdd.hAdd (HMul.hMul (↑α a) …
                        -/
              (continuous_snd.comp continuous_snd))))
                        /-
                          🎉 no goals
                        -/
                fun a b c => by rw [← coe_mul, ← coe_mul, ← coe_mul, ← coe_mul, mul_assoc]
    left_distrib := fun a b c =>
      Completion.induction_on₃ a b c
        (isClosed_eq
          (continuous_fst.mul
            (Continuous.add (continuous_fst.comp continuous_snd)
              (continuous_snd.comp continuous_snd)))
          (Continuous.add (continuous_fst.mul (continuous_fst.comp continuous_snd))
                        /-
                          α : Type u_1
                          inst✝³ : Ring α
                          inst✝² : UniformSpace α
                          inst✝¹ : TopologicalRing α
                          inst✝ : UniformAddGroup α
                          a✝ b✝ c✝ : UniformSpace.Completion α
                          a b c : α
                          ⊢ Eq (HMul.hMul (HAdd.hAdd (↑α a) (↑α b)) (↑α c)) (HAdd.hAdd (HMul.hMul (↑α a) …
                        -/
            (continuous_fst.mul (continuous_snd.comp continuous_snd))))
                        /-
                          🎉 no goals
                        -/
        fun a b c => by rw [← coe_add, ← coe_mul, ← coe_mul, ← coe_mul, ← coe_add, mul_add]
    right_distrib := fun a b c =>
      Completion.induction_on₃ a b c
        (isClosed_eq
          ((Continuous.add continuous_fst (continuous_fst.comp continuous_snd)).mul
            (continuous_snd.comp continuous_snd))
          (Continuous.add (continuous_fst.mul (continuous_snd.comp continuous_snd))
            ((continuous_fst.comp continuous_snd).mul
              (continuous_snd.comp continuous_snd))))
        fun a b c => by rw [← coe_add, ← coe_mul, ← coe_mul, ← coe_mul, ← coe_add, add_mul] }


/-- The map from a uniform ring to its completion, as a ring homomorphism. -/
def coeRingHom : α →+* Completion α where
  toFun := (↑)
  map_one' := coe_one α
  map_zero' := coe_zero
  map_add' := coe_add
  map_mul' := coe_mul


theorem continuous_coeRingHom : Continuous (coeRingHom : α → Completion α) :=
  continuous_coe α


/-- The completion extension as a ring morphism. -/
def extensionHom [CompleteSpace β] [T0Space β] : Completion α →+* β :=
  have hf' : Continuous (f : α →+ β) := hf
  -- helping the elaborator
  have hf : UniformContinuous f := uniformContinuous_addMonoidHom_of_continuous hf'
  { toFun := Completion.extension f
                    /-
                      α : Type u_1
                      inst✝⁹ : Ring α
                      inst✝⁸ : UniformSpace α
                      inst✝⁷ : TopologicalRing α
                      inst✝⁶ : UniformAddGroup α
                      β : Type u
                      inst✝⁵ : UniformSpace β
                      inst✝⁴ : Ring β
                      inst✝³ : UniformAddGroup β
                      inst✝² : TopologicalRing β
                      f : RingHom α β
                      hf✝ : Continuous ⇑f
                      inst✝¹ : CompleteSpace β
                      inst✝ : T0Space β
                      hf' : Continuous ⇑↑f
                      hf : UniformContinuous ⇑f
                      ⊢ Eq ((↑{ toFun := UniformSpace.Completion.extension ⇑f, map_one' := ⋯, map_mu …
                    -/
    map_zero' := by simp_rw [← coe_zero, extension_coe hf, f.map_zero]
                    /-
                      🎉 no goals
                    -/
    map_add' := fun a b =>
      Completion.induction_on₂ a b
        (isClosed_eq (continuous_extension.comp continuous_add)
          ((continuous_extension.comp continuous_fst).add
            (continuous_extension.comp continuous_snd)))
                   /-
                     α : Type u_1
                     inst✝⁹ : Ring α
                     inst✝⁸ : UniformSpace α
                     inst✝⁷ : TopologicalRing α
                     inst✝⁶ : UniformAddGroup α
                     β : Type u
                     inst✝⁵ : UniformSpace β
                     inst✝⁴ : Ring β
                     inst✝³ : UniformAddGroup β
                     inst✝² : TopologicalRing β
                     f : RingHom α β
                     hf✝ : Continuous ⇑f
                     inst✝¹ : CompleteSpace β
                     inst✝ : T0Space β
                     hf' : Continuous ⇑↑f
                     hf : UniformContinuous ⇑f
                     ⊢ Eq (UniformSpace.Completion.extension (⇑f) 1) 1
                   -/
        fun a b => by
                   /-
                     🎉 no goals
                   -/
        /-
          α : Type u_1
          inst✝⁹ : Ring α
          inst✝⁸ : UniformSpace α
          inst✝⁷ : TopologicalRing α
          inst✝⁶ : UniformAddGroup α
          β : Type u
          inst✝⁵ : UniformSpace β
          inst✝⁴ : Ring β
          inst✝³ : UniformAddGroup β
          inst✝² : TopologicalRing β
          f : RingHom α β
          hf✝ : Continuous ⇑f
          inst✝¹ : CompleteSpace β
          inst✝ : T0Space β
          hf' : Continuous ⇑↑f
          hf : UniformContinuous ⇑f
          a✝ b✝ : UniformSpace.Completion α
          a b : α
          ⊢ Eq ((↑{ toFun := UniformSpace.Completion.extension ⇑f, map_one' := ⋯, map_mu …
        -/
        simp_rw [← coe_add, extension_coe hf, f.map_add]
        /-
          🎉 no goals
        -/
    map_one' := by rw [← coe_one, extension_coe hf, f.map_one]
    map_mul' := fun a b =>
      Completion.induction_on₂ a b
        /-
          α : Type u_1
          inst✝⁹ : Ring α
          inst✝⁸ : UniformSpace α
          inst✝⁷ : TopologicalRing α
          inst✝⁶ : UniformAddGroup α
          β : Type u
          inst✝⁵ : UniformSpace β
          inst✝⁴ : Ring β
          inst✝³ : UniformAddGroup β
          inst✝² : TopologicalRing β
          f : RingHom α β
          hf✝ : Continuous ⇑f
          inst✝¹ : CompleteSpace β
          inst✝ : T0Space β
          hf' : Continuous ⇑↑f
          hf : UniformContinuous ⇑f
          a✝ b✝ : UniformSpace.Completion α
          a b : α
          ⊢ Eq ({ toFun := UniformSpace.Completion.extension ⇑f, map_one' := ⋯ }.toFun ( …
        -/
        (isClosed_eq (continuous_extension.comp continuous_mul)
        /-
          🎉 no goals
        -/
          ((continuous_extension.comp continuous_fst).mul
            (continuous_extension.comp continuous_snd)))
        fun a b => by
        simp_rw [← coe_mul, extension_coe hf, f.map_mul] }


theorem extensionHom_coe [CompleteSpace β] [T0Space β] (a : α) :
    Completion.extensionHom f hf a = f a := by
  simp only [Completion.extensionHom, RingHom.coe_mk, MonoidHom.coe_mk, OneHom.coe_mk,
    UniformSpace.Completion.extension_coe <| uniformContinuous_addMonoidHom_of_continuous hf]


instance topologicalRing : TopologicalRing (Completion α) where
  continuous_add := continuous_add
  continuous_mul := continuous_mul


/-- The completion map as a ring morphism. -/
def mapRingHom (hf : Continuous f) : Completion α →+* Completion β :=
  extensionHom (coeRingHom.comp f) (continuous_coeRingHom.comp hf)


@[simp]
theorem map_smul_eq_mul_coe (r : R) :
    Completion.map (r • ·) = ((algebraMap R A r : Completion A) * ·) := by
  /-
    A : Type u_2
    inst✝⁶ : Ring A
    inst✝⁵ : UniformSpace A
    inst✝⁴ : UniformAddGroup A
    inst✝³ : TopologicalRing A
    R : Type u_3
    inst✝² : CommSemiring R
    inst✝¹ : Algebra R A
    inst✝ : UniformContinuousConstSMul R A
    r : R
    ⊢ Eq (UniformSpace.Completion.map fun x => HSMul.hSMul r x) fun x => HMul.hMul …
  -/
  ext x
  /-
    case h
    A : Type u_2
    inst✝⁶ : Ring A
    inst✝⁵ : UniformSpace A
    inst✝⁴ : UniformAddGroup A
    inst✝³ : TopologicalRing A
    R : Type u_3
    inst✝² : CommSemiring R
    inst✝¹ : Algebra R A
    inst✝ : UniformContinuousConstSMul R A
    r : R
    x : UniformSpace.Completion A
    ⊢ Eq (UniformSpace.Completion.map (fun x => HSMul.hSMul r x) x) (HMul.hMul (↑A …
  -/
  refine Completion.induction_on x ?_ fun a => ?_
    /-
      case h.refine_1
      A : Type u_2
      inst✝⁶ : Ring A
      inst✝⁵ : UniformSpace A
      inst✝⁴ : UniformAddGroup A
      inst✝³ : TopologicalRing A
      R : Type u_3
      inst✝² : CommSemiring R
      inst✝¹ : Algebra R A
      inst✝ : UniformContinuousConstSMul R A
      r : R
      x : UniformSpace.Completion A
      ⊢ IsClosed (setOf fun a => Eq (UniformSpace.Completion.map (fun x => HSMul.hSM …
    -/
  · exact isClosed_eq Completion.continuous_map (continuous_mul_left _)
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      A : Type u_2
      inst✝⁶ : Ring A
      inst✝⁵ : UniformSpace A
      inst✝⁴ : UniformAddGroup A
      inst✝³ : TopologicalRing A
      R : Type u_3
      inst✝² : CommSemiring R
      inst✝¹ : Algebra R A
      inst✝ : UniformContinuousConstSMul R A
      r : R
      x : UniformSpace.Completion A
      a : A
      ⊢ Eq (UniformSpace.Completion.map (fun x => HSMul.hSMul r x) (↑A a)) (HMul.hMu …
    -/
  · simp_rw [map_coe (uniformContinuous_const_smul r) a, Algebra.smul_def, coe_mul]
    /-
      🎉 no goals
    -/


instance algebra : Algebra R (Completion A) :=
  { (UniformSpace.Completion.coeRingHom : A →+* Completion A).comp (algebraMap R A) with
    commutes' := fun r x =>
      Completion.induction_on x (isClosed_eq (continuous_mul_left _) (continuous_mul_right _))
        fun a => by
        /-
          α : Type u_1
          inst✝¹⁴ : Ring α
          inst✝¹³ : UniformSpace α
          inst✝¹² : TopologicalRing α
          inst✝¹¹ : UniformAddGroup α
          β : Type u
          inst✝¹⁰ : UniformSpace β
          inst✝⁹ : Ring β
          inst✝⁸ : UniformAddGroup β
          inst✝⁷ : TopologicalRing β
          f : RingHom α β
          hf : Continuous ⇑f
          A : Type u_2
          inst✝⁶ : Ring A
          inst✝⁵ : UniformSpace A
          inst✝⁴ : UniformAddGroup A
          inst✝³ : TopologicalRing A
          R : Type u_3
          inst✝² : CommSemiring R
          inst✝¹ : Algebra R A
          inst✝ : UniformContinuousConstSMul R A
          r : R
          x : UniformSpace.Completion A
          a : A
          ⊢ Eq (HMul.hMul (__src✝ r) (↑A a)) (HMul.hMul (↑A a) (__src✝ r))
        -/
        simpa only [coe_mul] using congr_arg ((↑) : A → Completion A) (Algebra.commutes r a)
        /-
          🎉 no goals
        -/
    smul_def' := fun r x => congr_fun (map_smul_eq_mul_coe A R r) x }


theorem algebraMap_def (r : R) :
    algebraMap R (Completion A) r = (algebraMap R A r : Completion A) :=
  rfl


instance commRing : CommRing (Completion R) :=
  { Completion.ring with
    mul_comm := fun a b =>
      Completion.induction_on₂ a b
        (isClosed_eq (continuous_fst.mul continuous_snd) (continuous_snd.mul continuous_fst))
                      /-
                        α : Type u_1
                        inst✝¹¹ : Ring α
                        inst✝¹⁰ : UniformSpace α
                        inst✝⁹ : TopologicalRing α
                        inst✝⁸ : UniformAddGroup α
                        β : Type u
                        inst✝⁷ : UniformSpace β
                        inst✝⁶ : Ring β
                        inst✝⁵ : UniformAddGroup β
                        inst✝⁴ : TopologicalRing β
                        f : RingHom α β
                        hf : Continuous ⇑f
                        R : Type u_2
                        inst✝³ : CommRing R
                        inst✝² : UniformSpace R
                        inst✝¹ : UniformAddGroup R
                        inst✝ : TopologicalRing R
                        a✝ b✝ : UniformSpace.Completion R
                        a b : R
                        ⊢ Eq (HMul.hMul (↑R a) (↑R b)) (HMul.hMul (↑R b) (↑R a))
                      -/
        fun a b => by rw [← coe_mul, ← coe_mul, mul_comm] }
                      /-
                        🎉 no goals
                      -/


/-- A shortcut instance for the common case -/
                                                   /-
                                                     α : Type u_1
                                                     inst✝¹¹ : Ring α
                                                     inst✝¹⁰ : UniformSpace α
                                                     inst✝⁹ : TopologicalRing α
                                                     inst✝⁸ : UniformAddGroup α
                                                     β : Type u
                                                     inst✝⁷ : UniformSpace β
                                                     inst✝⁶ : Ring β
                                                     inst✝⁵ : UniformAddGroup β
                                                     inst✝⁴ : TopologicalRing β
                                                     f : RingHom α β
                                                     hf : Continuous ⇑f
                                                     R : Type u_2
                                                     inst✝³ : CommRing R
                                                     inst✝² : UniformSpace R
                                                     inst✝¹ : UniformAddGroup R
                                                     inst✝ : TopologicalRing R
                                                     ⊢ Algebra R (UniformSpace.Completion R)
                                                   -/
instance algebra' : Algebra R (Completion R) := by infer_instance
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem inseparableSetoid_ring (α) [CommRing α] [TopologicalSpace α] [TopologicalRing α] :
    inseparableSetoid α = Submodule.quotientRel (Ideal.closure ⊥) :=
  Setoid.ext fun x y =>
                                                 /-
                                                   α : Type u_2
                                                   inst✝² : CommRing α
                                                   inst✝¹ : TopologicalSpace α
                                                   inst✝ : TopologicalRing α
                                                   x y : α
                                                   ⊢ Iff (Membership.mem (closure 0) (HSub.hSub x y)) (Membership.mem Bot.bot.clo …
                                                 -/
    addGroup_inseparable_iff.trans <| .trans (by rfl) (Submodule.quotientRel_def _).symm
                                                 /-
                                                   🎉 no goals
                                                 -/


@[deprecated (since := "2024-03-09")]
alias ring_sep_rel := inseparableSetoid_ring

-- Equality of types is evil

@[deprecated UniformSpace.inseparableSetoid_ring (since := "2024-02-16")]
theorem ring_sep_quot (α : Type u) [r : CommRing α] [TopologicalSpace α] [TopologicalRing α] :
    SeparationQuotient α = (α ⧸ (⊥ : Ideal α).closure) := by
  /-
    α : Type u
    r : CommRing α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalRing α
    ⊢ Eq (SeparationQuotient α) (HasQuotient.Quotient α Bot.bot.closure)
  -/
  rw [SeparationQuotient, @inseparableSetoid_ring α r]
  /-
    α : Type u
    r : CommRing α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalRing α
    ⊢ Eq (Quotient (Submodule.quotientRel Bot.bot.closure)) (HasQuotient.Quotient  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Given a topological ring `α` equipped with a uniform structure that makes subtraction uniformly
continuous, get an homeomorphism between the separated quotient of `α` and the quotient ring
corresponding to the closure of zero. -/
def sepQuotHomeomorphRingQuot (α) [CommRing α] [TopologicalSpace α] [TopologicalRing α] :
    SeparationQuotient α ≃ₜ α ⧸ (⊥ : Ideal α).closure where
                                               /-
                                                 α✝ : Type u_1
                                                 α : Type ?u.74293
                                                 inst✝² : CommRing α
                                                 inst✝¹ : TopologicalSpace α
                                                 inst✝ : TopologicalRing α
                                                 x y : α
                                                 ⊢ Iff ((inseparableSetoid α) x y) ((Submodule.quotientRel Bot.bot.closure) x y)
                                               -/
  toEquiv := Quotient.congrRight fun x y => by rw [inseparableSetoid_ring]
                                               /-
                                                 🎉 no goals
                                               -/
  continuous_toFun := continuous_id.quotient_map' <| by
    /-
      α✝ : Type u_1
      α : Type ?u.74293
      inst✝² : CommRing α
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalRing α
      ⊢ Relator.LiftFun (⇑(inseparableSetoid α)) (⇑(Submodule.quotientRel Bot.bot.cl …
    -/
    rw [inseparableSetoid_ring]; exact fun _ _ ↦ id
                                 /-
                                   🎉 no goals
                                 -/
  continuous_invFun := continuous_id.quotient_map' <| by
    /-
      α✝ : Type u_1
      α : Type ?u.74293
      inst✝² : CommRing α
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalRing α
      ⊢ Relator.LiftFun (⇑(Submodule.quotientRel Bot.bot.closure)) (⇑(inseparableSet …
    -/
    rw [inseparableSetoid_ring]; exact fun _ _ ↦ id
                                 /-
                                   🎉 no goals
                                 -/


instance commRing [CommRing α] [TopologicalSpace α] [TopologicalRing α] :
    CommRing (SeparationQuotient α) :=
  (sepQuotHomeomorphRingQuot _).commRing


/-- Given a topological ring `α` equipped with a uniform structure that makes subtraction uniformly
continuous, get an equivalence between the separated quotient of `α` and the quotient ring
corresponding to the closure of zero. -/
def sepQuotRingEquivRingQuot (α) [CommRing α] [TopologicalSpace α] [TopologicalRing α] :
    SeparationQuotient α ≃+* α ⧸ (⊥ : Ideal α).closure :=
  (sepQuotHomeomorphRingQuot _).ringEquiv


instance topologicalRing [CommRing α] [TopologicalSpace α] [TopologicalRing α] :
    TopologicalRing (SeparationQuotient α) where
  toContinuousAdd :=
    (sepQuotHomeomorphRingQuot α).isInducing.continuousAdd (sepQuotRingEquivRingQuot α)
  toContinuousMul :=
    (sepQuotHomeomorphRingQuot α).isInducing.continuousMul (sepQuotRingEquivRingQuot α)
  toContinuousNeg :=
    (sepQuotHomeomorphRingQuot α).isInducing.continuousNeg <|
      map_neg (sepQuotRingEquivRingQuot α)


/-- The dense inducing extension as a ring homomorphism. -/
noncomputable def IsDenseInducing.extendRingHom {i : α →+* β} {f : α →+* γ}
    (ue : IsUniformInducing i) (dr : DenseRange i) (hf : UniformContinuous f) : β →+* γ where
  toFun := (ue.isDenseInducing dr).extend f
  map_one' := by
    /-
      α : Type u_1
      inst✝⁹ : UniformSpace α
      inst✝⁸ : Semiring α
      β : Type u_2
      inst✝⁷ : UniformSpace β
      inst✝⁶ : Semiring β
      inst✝⁵ : TopologicalSemiring β
      γ : Type u_3
      inst✝⁴ : UniformSpace γ
      inst✝³ : Semiring γ
      inst✝² : TopologicalSemiring γ
      inst✝¹ : T2Space γ
      inst✝ : CompleteSpace γ
      i : RingHom α β
      f : RingHom α γ
      ue : IsUniformInducing ⇑i
      dr : DenseRange ⇑i
      hf : UniformContinuous ⇑f
      ⊢ Eq (⋯.extend (⇑f) 1) 1
    -/
    convert IsDenseInducing.extend_eq (ue.isDenseInducing dr) hf.continuous 1
    /-
      case h.e'_2.h.e'_10
      α : Type u_1
      inst✝⁹ : UniformSpace α
      inst✝⁸ : Semiring α
      β : Type u_2
      inst✝⁷ : UniformSpace β
      inst✝⁶ : Semiring β
      inst✝⁵ : TopologicalSemiring β
      γ : Type u_3
      inst✝⁴ : UniformSpace γ
      inst✝³ : Semiring γ
      inst✝² : TopologicalSemiring γ
      inst✝¹ : T2Space γ
      inst✝ : CompleteSpace γ
      i : RingHom α β
      f : RingHom α γ
      ue : IsUniformInducing ⇑i
      dr : DenseRange ⇑i
      hf : UniformContinuous ⇑f
      ⊢ Eq 1 (i 1)
    -/
    exacts [i.map_one.symm, f.map_one.symm]
    /-
      🎉 no goals
    -/
  map_zero' := by
    /-
      α : Type u_1
      inst✝⁹ : UniformSpace α
      inst✝⁸ : Semiring α
      β : Type u_2
      inst✝⁷ : UniformSpace β
      inst✝⁶ : Semiring β
      inst✝⁵ : TopologicalSemiring β
      γ : Type u_3
      inst✝⁴ : UniformSpace γ
      inst✝³ : Semiring γ
      inst✝² : TopologicalSemiring γ
      inst✝¹ : T2Space γ
      inst✝ : CompleteSpace γ
      i : RingHom α β
      f : RingHom α γ
      ue : IsUniformInducing ⇑i
      dr : DenseRange ⇑i
      hf : UniformContinuous ⇑f
      ⊢ Eq ((↑{ toFun := ⋯.extend ⇑f, map_one' := ⋯, map_mul' := ⋯ }).toFun 0) 0
    -/
    convert IsDenseInducing.extend_eq (ue.isDenseInducing dr) hf.continuous 0 <;>
    /-
      case h.e'_2.h.e'_1
      α : Type u_1
      inst✝⁹ : UniformSpace α
      inst✝⁸ : Semiring α
      β : Type u_2
      inst✝⁷ : UniformSpace β
      inst✝⁶ : Semiring β
      inst✝⁵ : TopologicalSemiring β
      γ : Type u_3
      inst✝⁴ : UniformSpace γ
      inst✝³ : Semiring γ
      inst✝² : TopologicalSemiring γ
      inst✝¹ : T2Space γ
      inst✝ : CompleteSpace γ
      i : RingHom α β
      f : RingHom α γ
      ue : IsUniformInducing ⇑i
      dr : DenseRange ⇑i
      hf : UniformContinuous ⇑f
      ⊢ Eq 0 (i 0)
    -/
    /-
      🎉 no goals
    -/
    simp only [map_zero]
    /-
      🎉 no goals
    -/
  map_add' := by
    /-
      α : Type u_1
      inst✝⁹ : UniformSpace α
      inst✝⁸ : Semiring α
      β : Type u_2
      inst✝⁷ : UniformSpace β
      inst✝⁶ : Semiring β
      inst✝⁵ : TopologicalSemiring β
      γ : Type u_3
      inst✝⁴ : UniformSpace γ
      inst✝³ : Semiring γ
      inst✝² : TopologicalSemiring γ
      inst✝¹ : T2Space γ
      inst✝ : CompleteSpace γ
      i : RingHom α β
      f : RingHom α γ
      ue : IsUniformInducing ⇑i
      dr : DenseRange ⇑i
      hf : UniformContinuous ⇑f
      ⊢ ∀ (x y : β), Eq ((↑{ toFun := ⋯.extend ⇑f, map_one' := ⋯, map_mul' := ⋯ }).t …
    -/
    have h := (uniformContinuous_uniformly_extend ue dr hf).continuous
    /-
      α : Type u_1
      inst✝⁹ : UniformSpace α
      inst✝⁸ : Semiring α
      β : Type u_2
      inst✝⁷ : UniformSpace β
      inst✝⁶ : Semiring β
      inst✝⁵ : TopologicalSemiring β
      γ : Type u_3
      inst✝⁴ : UniformSpace γ
      inst✝³ : Semiring γ
      inst✝² : TopologicalSemiring γ
      inst✝¹ : T2Space γ
      inst✝ : CompleteSpace γ
      i : RingHom α β
      f : RingHom α γ
      ue : IsUniformInducing ⇑i
      dr : DenseRange ⇑i
      hf : UniformContinuous ⇑f
      h : Continuous (⋯.extend ⇑f)
      ⊢ ∀ (x y : β), Eq ((↑{ toFun := ⋯.extend ⇑f, map_one' := ⋯, map_mul' := ⋯ }).t …
    -/
    /-
      α : Type u_1
      inst✝⁹ : UniformSpace α
      inst✝⁸ : Semiring α
      β : Type u_2
      inst✝⁷ : UniformSpace β
      inst✝⁶ : Semiring β
      inst✝⁵ : TopologicalSemiring β
      γ : Type u_3
      inst✝⁴ : UniformSpace γ
      inst✝³ : Semiring γ
      inst✝² : TopologicalSemiring γ
      inst✝¹ : T2Space γ
      inst✝ : CompleteSpace γ
      i : RingHom α β
      f : RingHom α γ
      ue : IsUniformInducing ⇑i
      dr : DenseRange ⇑i
      hf : UniformContinuous ⇑f
      ⊢ ∀ (x y : β), Eq ({ toFun := ⋯.extend ⇑f, map_one' := ⋯ }.toFun (HMul.hMul x  …
    -/
    refine fun x y => DenseRange.induction_on₂ dr ?_ (fun a b => ?_) x y
    /-
      α : Type u_1
      inst✝⁹ : UniformSpace α
      inst✝⁸ : Semiring α
      β : Type u_2
      inst✝⁷ : UniformSpace β
      inst✝⁶ : Semiring β
      inst✝⁵ : TopologicalSemiring β
      γ : Type u_3
      inst✝⁴ : UniformSpace γ
      inst✝³ : Semiring γ
      inst✝² : TopologicalSemiring γ
      inst✝¹ : T2Space γ
      inst✝ : CompleteSpace γ
      i : RingHom α β
      f : RingHom α γ
      ue : IsUniformInducing ⇑i
      dr : DenseRange ⇑i
      hf : UniformContinuous ⇑f
      h : Continuous (⋯.extend ⇑f)
      ⊢ ∀ (x y : β), Eq ({ toFun := ⋯.extend ⇑f, map_one' := ⋯ }.toFun (HMul.hMul x  …
    -/
    · exact isClosed_eq (Continuous.comp h continuous_add)
        ((h.comp continuous_fst).add (h.comp continuous_snd))
    · simp_rw [← i.map_add, IsDenseInducing.extend_eq (ue.isDenseInducing dr) hf.continuous _,
        ← f.map_add]
  map_mul' := by
    have h := (uniformContinuous_uniformly_extend ue dr hf).continuous
    refine fun x y => DenseRange.induction_on₂ dr ?_ (fun a b => ?_) x y
    · exact isClosed_eq (Continuous.comp h continuous_mul)
        ((h.comp continuous_fst).mul (h.comp continuous_snd))
    · simp_rw [← i.map_mul, IsDenseInducing.extend_eq (ue.isDenseInducing dr) hf.continuous _,
        ← f.map_mul]


