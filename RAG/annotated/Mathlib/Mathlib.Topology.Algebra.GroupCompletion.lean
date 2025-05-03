instance [Zero α] : Zero (Completion α) :=
  ⟨(0 : α)⟩


instance [Neg α] : Neg (Completion α) :=
  ⟨Completion.map (fun a ↦ -a : α → α)⟩


instance [Add α] : Add (Completion α) :=
  ⟨Completion.map₂ (· + ·)⟩


instance [Sub α] : Sub (Completion α) :=
  ⟨Completion.map₂ Sub.sub⟩


@[norm_cast]
theorem UniformSpace.Completion.coe_zero [Zero α] : ((0 : α) : Completion α) = 0 :=
  rfl


instance [UniformSpace α] [MonoidWithZero M] [Zero α] [MulActionWithZero M α]
    [UniformContinuousConstSMul M α] : MulActionWithZero M (Completion α) :=
  { (inferInstance : MulAction M <| Completion α) with
                            /-
                              M : Type u_1
                              R : Type u_2
                              α : Type u_3
                              β : Type u_4
                              inst✝⁴ : UniformSpace α
                              inst✝³ : MonoidWithZero M
                              inst✝² : Zero α
                              inst✝¹ : MulActionWithZero M α
                              inst✝ : UniformContinuousConstSMul M α
                              r : M
                              ⊢ Eq (HSMul.hSMul r 0) 0
                            -/
    smul_zero := fun r ↦ by rw [← coe_zero, ← coe_smul, MulActionWithZero.smul_zero r]
                            /-
                              🎉 no goals
                            -/
    zero_smul :=
      ext' (continuous_const_smul _) continuous_const fun a ↦ by
        /-
          M : Type u_1
          R : Type u_2
          α : Type u_3
          β : Type u_4
          inst✝⁴ : UniformSpace α
          inst✝³ : MonoidWithZero M
          inst✝² : Zero α
          inst✝¹ : MulActionWithZero M α
          inst✝ : UniformContinuousConstSMul M α
          a : α
          ⊢ Eq (HSMul.hSMul 0 (↑α a)) 0
        -/
        rw [← coe_smul, zero_smul, coe_zero] }
        /-
          🎉 no goals
        -/


@[norm_cast]
theorem coe_neg (a : α) : ((-a : α) : Completion α) = -a :=
  (map_coe uniformContinuous_neg a).symm


@[norm_cast]
theorem coe_sub (a b : α) : ((a - b : α) : Completion α) = a - b :=
  (map₂_coe_coe a b Sub.sub uniformContinuous_sub).symm


@[norm_cast]
theorem coe_add (a b : α) : ((a + b : α) : Completion α) = a + b :=
  (map₂_coe_coe a b (· + ·) uniformContinuous_add).symm


instance : AddMonoid (Completion α) :=
  { (inferInstance : Zero <| Completion α),
    (inferInstance : Add <| Completion α) with
    zero_add := fun a ↦
      Completion.induction_on a
        (isClosed_eq (continuous_map₂ continuous_const continuous_id) continuous_id) fun a ↦
                                           /-
                                             M : Type u_1
                                             R : Type u_2
                                             α : Type u_3
                                             β : Type u_4
                                             inst✝² : UniformSpace α
                                             inst✝¹ : AddGroup α
                                             inst✝ : UniformAddGroup α
                                             a✝ : UniformSpace.Completion α
                                             a : α
                                             ⊢ Eq (HAdd.hAdd 0 (↑α a)) (↑α a)
                                           -/
        show 0 + (a : Completion α) = a by rw [← coe_zero, ← coe_add, zero_add]
                                           /-
                                             🎉 no goals
                                           -/
    add_zero := fun a ↦
      Completion.induction_on a
        (isClosed_eq (continuous_map₂ continuous_id continuous_const) continuous_id) fun a ↦
                                           /-
                                             M : Type u_1
                                             R : Type u_2
                                             α : Type u_3
                                             β : Type u_4
                                             inst✝² : UniformSpace α
                                             inst✝¹ : AddGroup α
                                             inst✝ : UniformAddGroup α
                                             a✝ : UniformSpace.Completion α
                                             a : α
                                             ⊢ Eq (HAdd.hAdd (↑α a) 0) (↑α a)
                                           -/
        show (a : Completion α) + 0 = a by rw [← coe_zero, ← coe_add, add_zero]
                                           /-
                                             🎉 no goals
                                           -/
    add_assoc := fun a b c ↦
      Completion.induction_on₃ a b c
        (isClosed_eq
          (continuous_map₂ (continuous_map₂ continuous_fst (continuous_fst.comp continuous_snd))
            (continuous_snd.comp continuous_snd))
                                                         /-
                                                           M : Type u_1
                                                           R : Type u_2
                                                           α : Type u_3
                                                           β : Type u_4
                                                           inst✝² : UniformSpace α
                                                           inst✝¹ : AddGroup α
                                                           inst✝ : UniformAddGroup α
                                                           a✝ b✝ c✝ : UniformSpace.Completion α
                                                           a b c : α
                                                           ⊢ Eq (HAdd.hAdd (HAdd.hAdd (↑α a) (↑α b)) (↑α c)) (HAdd.hAdd (↑α a) (HAdd.hAdd …
                                                         -/
          (continuous_map₂ continuous_fst
                                                         /-
                                                           🎉 no goals
                                                         -/
            (continuous_map₂ (continuous_fst.comp continuous_snd)
              (continuous_snd.comp continuous_snd))))
        fun a b c ↦
        show (a : Completion α) + b + c = a + (b + c) by repeat' rw_mod_cast [add_assoc]
    nsmul := (· • ·)
    nsmul_zero := fun a ↦
      Completion.induction_on a (isClosed_eq continuous_map continuous_const) fun a ↦
                                           /-
                                             M : Type u_1
                                             R : Type u_2
                                             α : Type u_3
                                             β : Type u_4
                                             inst✝² : UniformSpace α
                                             inst✝¹ : AddGroup α
                                             inst✝ : UniformAddGroup α
                                             a✝ : UniformSpace.Completion α
                                             a : α
                                             ⊢ Eq (HSMul.hSMul 0 (↑α a)) 0
                                           -/
        show 0 • (a : Completion α) = 0 by rw [← coe_smul, ← coe_zero, zero_smul]
                                           /-
                                             🎉 no goals
                                           -/
    nsmul_succ := fun n a ↦
      Completion.induction_on a
        (isClosed_eq continuous_map <| continuous_map₂ continuous_map continuous_id) fun a ↦
        show (n + 1) • (a : Completion α) = n • (a : Completion α) + (a : Completion α) by
          /-
            M : Type u_1
            R : Type u_2
            α : Type u_3
            β : Type u_4
            inst✝² : UniformSpace α
            inst✝¹ : AddGroup α
            inst✝ : UniformAddGroup α
            n : Nat
            a✝ : UniformSpace.Completion α
            a : α
            ⊢ Eq (HSMul.hSMul (HAdd.hAdd n 1) (↑α a)) (HAdd.hAdd (HSMul.hSMul n (↑α a)) (↑ …
          -/
          rw [← coe_smul, succ_nsmul, coe_add, coe_smul] }
          /-
            🎉 no goals
          -/


instance : SubNegMonoid (Completion α) :=
  { (inferInstance : AddMonoid <| Completion α),
    (inferInstance : Neg <| Completion α),
    (inferInstance : Sub <| Completion α) with
    sub_eq_add_neg := fun a b ↦
      Completion.induction_on₂ a b
        (isClosed_eq (continuous_map₂ continuous_fst continuous_snd)
          (continuous_map₂ continuous_fst (Completion.continuous_map.comp continuous_snd)))
        fun a b ↦ mod_cast congr_arg ((↑) : α → Completion α) (sub_eq_add_neg a b)
    zsmul := (· • ·)
    zsmul_zero' := fun a ↦
      Completion.induction_on a (isClosed_eq continuous_map continuous_const) fun a ↦
                                                 /-
                                                   M : Type u_1
                                                   R : Type u_2
                                                   α : Type u_3
                                                   β : Type u_4
                                                   inst✝² : UniformSpace α
                                                   inst✝¹ : AddGroup α
                                                   inst✝ : UniformAddGroup α
                                                   a✝ : UniformSpace.Completion α
                                                   a : α
                                                   ⊢ Eq (HSMul.hSMul 0 (↑α a)) 0
                                                 -/
        show (0 : ℤ) • (a : Completion α) = 0 by rw [← coe_smul, ← coe_zero, zero_smul]
                                                 /-
                                                   🎉 no goals
                                                 -/
    zsmul_succ' := fun n a ↦
      Completion.induction_on a
        (isClosed_eq continuous_map <| continuous_map₂ continuous_map continuous_id) fun a ↦
          show (n.succ : ℤ) • (a : Completion α) = _ by
            rw [← coe_smul, show (n.succ : ℤ) • a = (n : ℤ) • a + a from
              SubNegMonoid.zsmul_succ' n a, coe_add, coe_smul]
    zsmul_neg' := fun n a ↦
      Completion.induction_on a
        (isClosed_eq continuous_map <| Completion.continuous_map.comp continuous_map) fun a ↦
          show (Int.negSucc n) • (a : Completion α) = _ by
            rw [← coe_smul, show (Int.negSucc n) • a = -((n.succ : ℤ) • a) from
              SubNegMonoid.zsmul_neg' n a, coe_neg, coe_smul] }


instance addGroup : AddGroup (Completion α) :=
  { (inferInstance : SubNegMonoid <| Completion α) with
    neg_add_cancel := fun a ↦
      Completion.induction_on a
        (isClosed_eq (continuous_map₂ Completion.continuous_map continuous_id) continuous_const)
        fun a ↦
        show -(a : Completion α) + a = 0 by
          /-
            M : Type u_1
            R : Type u_2
            α : Type u_3
            β : Type u_4
            inst✝² : UniformSpace α
            inst✝¹ : AddGroup α
            inst✝ : UniformAddGroup α
            a✝ : UniformSpace.Completion α
            a : α
            ⊢ Eq (HAdd.hAdd (Neg.neg (↑α a)) (↑α a)) 0
          -/
          rw_mod_cast [neg_add_cancel]
          /-
            M : Type u_1
            R : Type u_2
            α : Type u_3
            β : Type u_4
            inst✝² : UniformSpace α
            inst✝¹ : AddGroup α
            inst✝ : UniformAddGroup α
            a✝ : UniformSpace.Completion α
            a : α
            ⊢ Eq (↑α 0) 0
          -/
          rfl }
          /-
            🎉 no goals
          -/


instance uniformAddGroup : UniformAddGroup (Completion α) :=
  ⟨uniformContinuous_map₂ Sub.sub⟩


instance {M} [Monoid M] [DistribMulAction M α] [UniformContinuousConstSMul M α] :
    DistribMulAction M (Completion α) :=
  { (inferInstance : MulAction M <| Completion α) with
    smul_add := fun r x y ↦
      induction_on₂ x y
        (isClosed_eq ((continuous_fst.add continuous_snd).const_smul _)
          ((continuous_fst.const_smul _).add (continuous_snd.const_smul _)))
                     /-
                       M✝ : Type u_1
                       R : Type u_2
                       α : Type u_3
                       β : Type u_4
                       inst✝⁵ : UniformSpace α
                       inst✝⁴ : AddGroup α
                       inst✝³ : UniformAddGroup α
                       M : Type ?u.35875
                       inst✝² : Monoid M
                       inst✝¹ : DistribMulAction M α
                       inst✝ : UniformContinuousConstSMul M α
                       r : M
                       x y : UniformSpace.Completion α
                       a b : α
                       ⊢ Eq (HSMul.hSMul r (HAdd.hAdd (↑α a) (↑α b))) (HAdd.hAdd (HSMul.hSMul r (↑α a …
                     -/
                            /-
                              M✝ : Type u_1
                              R : Type u_2
                              α : Type u_3
                              β : Type u_4
                              inst✝⁵ : UniformSpace α
                              inst✝⁴ : AddGroup α
                              inst✝³ : UniformAddGroup α
                              M : Type ?u.35875
                              inst✝² : Monoid M
                              inst✝¹ : DistribMulAction M α
                              inst✝ : UniformContinuousConstSMul M α
                              r : M
                              ⊢ Eq (HSMul.hSMul r 0) 0
                            -/
        fun a b ↦ by simp only [← coe_add, ← coe_smul, smul_add]
                            /-
                              🎉 no goals
                            -/
                     /-
                       🎉 no goals
                     -/
    smul_zero := fun r ↦ by rw [← coe_zero, ← coe_smul, smul_zero r] }


/-- The map from a group to its completion as a group hom. -/
@[simps]
def toCompl : α →+ Completion α where
  toFun := (↑)
  map_add' := coe_add
  map_zero' := coe_zero


theorem continuous_toCompl : Continuous (toCompl : α → Completion α) :=
  continuous_coe α


theorem isDenseInducing_toCompl : IsDenseInducing (toCompl : α → Completion α) :=
  isDenseInducing_coe


instance instAddCommGroup : AddCommGroup (Completion α) :=
  { (inferInstance : AddGroup <| Completion α) with
    add_comm := fun a b ↦
      Completion.induction_on₂ a b
        (isClosed_eq (continuous_map₂ continuous_fst continuous_snd)
          (continuous_map₂ continuous_snd continuous_fst))
        fun x y ↦ by
        /-
          M : Type u_1
          R : Type u_2
          α : Type u_3
          β : Type u_4
          inst✝² : UniformSpace α
          inst✝¹ : AddCommGroup α
          inst✝ : UniformAddGroup α
          a b : UniformSpace.Completion α
          x y : α
          ⊢ Eq (HAdd.hAdd (↑α x) (↑α y)) (HAdd.hAdd (↑α y) (↑α x))
        -/
        change (x : Completion α) + ↑y = ↑y + ↑x
        /-
          M : Type u_1
          R : Type u_2
          α : Type u_3
          β : Type u_4
          inst✝² : UniformSpace α
          inst✝¹ : AddCommGroup α
          inst✝ : UniformAddGroup α
          a b : UniformSpace.Completion α
          x y : α
          ⊢ Eq (HAdd.hAdd (↑α x) (↑α y)) (HAdd.hAdd (↑α y) (↑α x))
        -/
        rw [← coe_add, ← coe_add, add_comm] }
        /-
          🎉 no goals
        -/


instance instModule [Semiring R] [Module R α] [UniformContinuousConstSMul R α] :
    Module R (Completion α) :=
  { (inferInstance : DistribMulAction R <| Completion α),
    (inferInstance : MulActionWithZero R <| Completion α) with
    add_smul := fun a b ↦
      ext' (continuous_const_smul _) ((continuous_const_smul _).add (continuous_const_smul _))
        fun x ↦ by
          /-
            M : Type u_1
            R : Type u_2
            α : Type u_3
            β : Type u_4
            inst✝⁵ : UniformSpace α
            inst✝⁴ : AddCommGroup α
            inst✝³ : UniformAddGroup α
            inst✝² : Semiring R
            inst✝¹ : Module R α
            inst✝ : UniformContinuousConstSMul R α
            a b : R
            x : α
            ⊢ Eq (HSMul.hSMul (HAdd.hAdd a b) (↑α x)) (HAdd.hAdd (HSMul.hSMul a (↑α x)) (H …
          -/
          rw [← coe_smul, add_smul, coe_add, coe_smul, coe_smul] }
          /-
            🎉 no goals
          -/


/-- Extension to the completion of a continuous group hom. -/
def AddMonoidHom.extension [CompleteSpace β] [T0Space β] (f : α →+ β) (hf : Continuous f) :
    Completion α →+ β :=
  have hf : UniformContinuous f := uniformContinuous_addMonoidHom_of_continuous hf
  { toFun := Completion.extension f
                    /-
                      M : Type u_1
                      R : Type u_2
                      α : Type u_3
                      β : Type u_4
                      inst✝⁷ : UniformSpace α
                      inst✝⁶ : AddGroup α
                      inst✝⁵ : UniformAddGroup α
                      inst✝⁴ : UniformSpace β
                      inst✝³ : AddGroup β
                      inst✝² : UniformAddGroup β
                      inst✝¹ : CompleteSpace β
                      inst✝ : T0Space β
                      f : AddMonoidHom α β
                      hf✝ : Continuous ⇑f
                      hf : UniformContinuous ⇑f
                      ⊢ Eq (UniformSpace.Completion.extension (⇑f) 0) 0
                    -/
    map_zero' := by rw [← coe_zero, extension_coe hf, f.map_zero]
                    /-
                      🎉 no goals
                    -/
    map_add' := fun a b ↦
      Completion.induction_on₂ a b
        (isClosed_eq (continuous_extension.comp continuous_add)
          ((continuous_extension.comp continuous_fst).add
            (continuous_extension.comp continuous_snd)))
        fun a b ↦
        show Completion.extension f _ = Completion.extension f _ + Completion.extension f _ by
        /-
          M : Type u_1
          R : Type u_2
          α : Type u_3
          β : Type u_4
          inst✝⁷ : UniformSpace α
          inst✝⁶ : AddGroup α
          inst✝⁵ : UniformAddGroup α
          inst✝⁴ : UniformSpace β
          inst✝³ : AddGroup β
          inst✝² : UniformAddGroup β
          inst✝¹ : CompleteSpace β
          inst✝ : T0Space β
          f : AddMonoidHom α β
          hf✝ : Continuous ⇑f
          hf : UniformContinuous ⇑f
          a✝ b✝ : UniformSpace.Completion α
          a b : α
          ⊢ Eq (UniformSpace.Completion.extension (⇑f) (HAdd.hAdd (↑α a) (↑α b))) (HAdd. …
        -/
        rw_mod_cast [extension_coe hf, extension_coe hf, extension_coe hf, f.map_add] }
        /-
          🎉 no goals
        -/


theorem AddMonoidHom.extension_coe [CompleteSpace β] [T0Space β] (f : α →+ β)
    (hf : Continuous f) (a : α) : f.extension hf a = f a :=
  UniformSpace.Completion.extension_coe (uniformContinuous_addMonoidHom_of_continuous hf) a


@[continuity]
theorem AddMonoidHom.continuous_extension [CompleteSpace β] [T0Space β] (f : α →+ β)
    (hf : Continuous f) : Continuous (f.extension hf) :=
  UniformSpace.Completion.continuous_extension


/-- Completion of a continuous group hom, as a group hom. -/
def AddMonoidHom.completion (f : α →+ β) (hf : Continuous f) : Completion α →+ Completion β :=
  (toCompl.comp f).extension (continuous_toCompl.comp hf)


@[continuity]
theorem AddMonoidHom.continuous_completion (f : α →+ β) (hf : Continuous f) :
    Continuous (AddMonoidHom.completion f hf : Completion α → Completion β) :=
  continuous_map


theorem AddMonoidHom.completion_coe (f : α →+ β) (hf : Continuous f) (a : α) :
    AddMonoidHom.completion f hf a = f a :=
  map_coe (uniformContinuous_addMonoidHom_of_continuous hf) a


theorem AddMonoidHom.completion_zero :
    AddMonoidHom.completion (0 : α →+ β) continuous_const = 0 := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝⁵ : UniformSpace α
    inst✝⁴ : AddGroup α
    inst✝³ : UniformAddGroup α
    inst✝² : UniformSpace β
    inst✝¹ : AddGroup β
    inst✝ : UniformAddGroup β
    ⊢ Eq (AddMonoidHom.completion 0 ⋯) 0
  -/
  ext x
  /-
    case h
    α : Type u_3
    β : Type u_4
    inst✝⁵ : UniformSpace α
    inst✝⁴ : AddGroup α
    inst✝³ : UniformAddGroup α
    inst✝² : UniformSpace β
    inst✝¹ : AddGroup β
    inst✝ : UniformAddGroup β
    x : UniformSpace.Completion α
    ⊢ Eq ((AddMonoidHom.completion 0 ⋯) x) (0 x)
  -/
  refine Completion.induction_on x ?_ ?_
    /-
      case h.refine_1
      α : Type u_3
      β : Type u_4
      inst✝⁵ : UniformSpace α
      inst✝⁴ : AddGroup α
      inst✝³ : UniformAddGroup α
      inst✝² : UniformSpace β
      inst✝¹ : AddGroup β
      inst✝ : UniformAddGroup β
      x : UniformSpace.Completion α
      ⊢ IsClosed (setOf fun a => Eq ((AddMonoidHom.completion 0 ⋯) a) (0 a))
    -/
  · apply isClosed_eq (AddMonoidHom.continuous_completion (0 : α →+ β) continuous_const)
    /-
      case h.refine_1
      α : Type u_3
      β : Type u_4
      inst✝⁵ : UniformSpace α
      inst✝⁴ : AddGroup α
      inst✝³ : UniformAddGroup α
      inst✝² : UniformSpace β
      inst✝¹ : AddGroup β
      inst✝ : UniformAddGroup β
      x : UniformSpace.Completion α
      ⊢ Continuous ⇑0
    -/
    exact continuous_const
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      α : Type u_3
      β : Type u_4
      inst✝⁵ : UniformSpace α
      inst✝⁴ : AddGroup α
      inst✝³ : UniformAddGroup α
      inst✝² : UniformSpace β
      inst✝¹ : AddGroup β
      inst✝ : UniformAddGroup β
      x : UniformSpace.Completion α
      ⊢ ∀ (a : α), Eq ((AddMonoidHom.completion 0 ⋯) (↑α a)) (0 (↑α a))
    -/
  · intro a
    /-
      case h.refine_2
      α : Type u_3
      β : Type u_4
      inst✝⁵ : UniformSpace α
      inst✝⁴ : AddGroup α
      inst✝³ : UniformAddGroup α
      inst✝² : UniformSpace β
      inst✝¹ : AddGroup β
      inst✝ : UniformAddGroup β
      x : UniformSpace.Completion α
      a : α
      ⊢ Eq ((AddMonoidHom.completion 0 ⋯) (↑α a)) (0 (↑α a))
    -/
    simp [(0 : α →+ β).completion_coe continuous_const, coe_zero]
    /-
      🎉 no goals
    -/


theorem AddMonoidHom.completion_add {γ : Type*} [AddCommGroup γ] [UniformSpace γ]
    [UniformAddGroup γ] (f g : α →+ γ) (hf : Continuous f) (hg : Continuous g) :
    AddMonoidHom.completion (f + g) (hf.add hg) =
    AddMonoidHom.completion f hf + AddMonoidHom.completion g hg := by
  /-
    α : Type u_3
    inst✝⁵ : UniformSpace α
    inst✝⁴ : AddGroup α
    inst✝³ : UniformAddGroup α
    γ : Type u_5
    inst✝² : AddCommGroup γ
    inst✝¹ : UniformSpace γ
    inst✝ : UniformAddGroup γ
    f g : AddMonoidHom α γ
    hf : Continuous ⇑f
    hg : Continuous ⇑g
    ⊢ Eq ((HAdd.hAdd f g).completion ⋯) (HAdd.hAdd (f.completion hf) (g.completion …
  -/
  have hfg := hf.add hg
  /-
    α : Type u_3
    inst✝⁵ : UniformSpace α
    inst✝⁴ : AddGroup α
    inst✝³ : UniformAddGroup α
    γ : Type u_5
    inst✝² : AddCommGroup γ
    inst✝¹ : UniformSpace γ
    inst✝ : UniformAddGroup γ
    f g : AddMonoidHom α γ
    hf : Continuous ⇑f
    hg : Continuous ⇑g
    hfg : Continuous fun x => HAdd.hAdd (f x) (g x)
    ⊢ Eq ((HAdd.hAdd f g).completion ⋯) (HAdd.hAdd (f.completion hf) (g.completion …
  -/
  ext x
  /-
    case h
    α : Type u_3
    inst✝⁵ : UniformSpace α
    inst✝⁴ : AddGroup α
    inst✝³ : UniformAddGroup α
    γ : Type u_5
    inst✝² : AddCommGroup γ
    inst✝¹ : UniformSpace γ
    inst✝ : UniformAddGroup γ
    f g : AddMonoidHom α γ
    hf : Continuous ⇑f
    hg : Continuous ⇑g
    hfg : Continuous fun x => HAdd.hAdd (f x) (g x)
    x : UniformSpace.Completion α
    ⊢ Eq (((HAdd.hAdd f g).completion ⋯) x) ((HAdd.hAdd (f.completion hf) (g.compl …
  -/
  refine Completion.induction_on x ?_ ?_
  · exact isClosed_eq ((f + g).continuous_completion hfg)
      ((f.continuous_completion hf).add (g.continuous_completion hg))
    /-
      case h.refine_2
      α : Type u_3
      inst✝⁵ : UniformSpace α
      inst✝⁴ : AddGroup α
      inst✝³ : UniformAddGroup α
      γ : Type u_5
      inst✝² : AddCommGroup γ
      inst✝¹ : UniformSpace γ
      inst✝ : UniformAddGroup γ
      f g : AddMonoidHom α γ
      hf : Continuous ⇑f
      hg : Continuous ⇑g
      hfg : Continuous fun x => HAdd.hAdd (f x) (g x)
      x : UniformSpace.Completion α
      ⊢ ∀ (a : α), Eq (((HAdd.hAdd f g).completion ⋯) (↑α a)) ((HAdd.hAdd (f.complet …
    -/
  · intro a
    /-
      case h.refine_2
      α : Type u_3
      inst✝⁵ : UniformSpace α
      inst✝⁴ : AddGroup α
      inst✝³ : UniformAddGroup α
      γ : Type u_5
      inst✝² : AddCommGroup γ
      inst✝¹ : UniformSpace γ
      inst✝ : UniformAddGroup γ
      f g : AddMonoidHom α γ
      hf : Continuous ⇑f
      hg : Continuous ⇑g
      hfg : Continuous fun x => HAdd.hAdd (f x) (g x)
      x : UniformSpace.Completion α
      a : α
      ⊢ Eq (((HAdd.hAdd f g).completion ⋯) (↑α a)) ((HAdd.hAdd (f.completion hf) (g. …
    -/
    simp [(f + g).completion_coe hfg, coe_add, f.completion_coe hf, g.completion_coe hg]
    /-
      🎉 no goals
    -/


