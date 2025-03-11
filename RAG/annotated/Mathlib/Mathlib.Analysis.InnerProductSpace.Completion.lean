theorem Inseparable.inner_eq_inner {x₁ x₂ y₁ y₂ : E}
    (hx : Inseparable x₁ x₂) (hy : Inseparable y₁ y₂) :
    inner x₁ y₁ = (inner x₂ y₂ : 𝕜) :=
  ((hx.prod hy).map continuous_inner).eq


instance : Inner 𝕜 (SeparationQuotient E) where
  inner := SeparationQuotient.lift₂ Inner.inner fun _ _ _ _ => Inseparable.inner_eq_inner


@[simp]
theorem inner_mk_mk (x y : E) :
    inner (mk x) (mk y) = (inner x y : 𝕜) := rfl


instance : InnerProductSpace 𝕜 (SeparationQuotient E) where
  norm_sq_eq_inner := Quotient.ind norm_sq_eq_inner
  conj_symm := Quotient.ind₂ inner_conj_symm
  add_left := Quotient.ind fun x => Quotient.ind₂ <| inner_add_left x
  smul_left := Quotient.ind₂ inner_smul_left


local notation "⟪" x ", " y "⟫" => @inner 𝕜 _ _ x y


local notation "IK" => @RCLike.I 𝕜 _


local postfix:90 "†" => starRingEnd _


instance toInner {𝕜' E' : Type*} [TopologicalSpace 𝕜'] [UniformSpace E'] [Inner 𝕜' E'] :
    Inner 𝕜' (Completion E') where
  inner := curry <| (isDenseInducing_coe.prodMap isDenseInducing_coe).extend (uncurry inner)


@[simp]
theorem inner_coe (a b : E) : inner (a : Completion E) (b : Completion E) = (inner a b : 𝕜) :=
  (isDenseInducing_coe.prodMap isDenseInducing_coe).extend_eq
    (continuous_inner : Continuous (uncurry inner : E × E → 𝕜)) (a, b)


protected theorem continuous_inner :
    Continuous (uncurry inner : Completion E × Completion E → 𝕜) := by
  let inner' : E →+ E →+ 𝕜 :=
    { toFun := fun x => (innerₛₗ 𝕜 x).toAddMonoidHom
      map_zero' := by ext x; exact inner_zero_left _
      map_add' := fun x y => by ext z; exact inner_add_left _ _ _ }
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    inner' : AddMonoidHom E (AddMonoidHom E 𝕜) := { toFun := fun x => ((innerₛₗ 𝕜) …
    ⊢ Continuous (Function.uncurry Inner.inner)
  -/
  have : Continuous fun p : E × E => inner' p.1 p.2 := continuous_inner
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    inner' : AddMonoidHom E (AddMonoidHom E 𝕜) := { toFun := fun x => ((innerₛₗ 𝕜) …
    this : Continuous fun p => (inner' p.1) p.2
    ⊢ Continuous (Function.uncurry Inner.inner)
  -/
  rw [Completion.toInner, inner, uncurry_curry _]
  change
    Continuous
      (((isDenseInducing_toCompl E).prodMap (isDenseInducing_toCompl E)).extend fun p : E × E =>
        inner' p.1 p.2)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    inner' : AddMonoidHom E (AddMonoidHom E 𝕜) := { toFun := fun x => ((innerₛₗ 𝕜) …
    this : Continuous fun p => (inner' p.1) p.2
    ⊢ Continuous (⋯.extend fun p => (inner' p.1) p.2)
  -/
  exact (isDenseInducing_toCompl E).extend_Z_bilin (isDenseInducing_toCompl E) this
  /-
    🎉 no goals
  -/


protected theorem Continuous.inner {α : Type*} [TopologicalSpace α] {f g : α → Completion E}
    (hf : Continuous f) (hg : Continuous g) : Continuous (fun x : α => inner (f x) (g x) : α → 𝕜) :=
  UniformSpace.Completion.continuous_inner.comp (hf.prod_mk hg : _)


instance innerProductSpace : InnerProductSpace 𝕜 (Completion E) where
  norm_sq_eq_inner x :=
    Completion.induction_on x
      (isClosed_eq (continuous_norm.pow 2)
        (continuous_re.comp (Continuous.inner continuous_id' continuous_id')))
                  /-
                    𝕜 : Type u_1
                    E : Type u_2
                    F : Type u_3
                    inst✝² : RCLike 𝕜
                    inst✝¹ : SeminormedAddCommGroup E
                    inst✝ : InnerProductSpace 𝕜 E
                    x : UniformSpace.Completion E
                    a : E
                    ⊢ Eq (HPow.hPow (Norm.norm (↑E a)) 2) (RCLike.re (Inner.inner (↑E a) (↑E a)))
                  -/
      fun a => by simp only [norm_coe, inner_coe, inner_self_eq_norm_sq]
                  /-
                    🎉 no goals
                  -/
  conj_symm x y :=
    Completion.induction_on₂ x y
      (isClosed_eq (continuous_conj.comp (Continuous.inner continuous_snd continuous_fst))
        (Continuous.inner continuous_fst continuous_snd))
                    /-
                      𝕜 : Type u_1
                      E : Type u_2
                      F : Type u_3
                      inst✝² : RCLike 𝕜
                      inst✝¹ : SeminormedAddCommGroup E
                      inst✝ : InnerProductSpace 𝕜 E
                      x y : UniformSpace.Completion E
                      a b : E
                      ⊢ Eq ((starRingEnd 𝕜) (Inner.inner (↑E b) (↑E a))) (Inner.inner (↑E a) (↑E b))
                    -/
      fun a b => by simp only [inner_coe, inner_conj_symm]
                    /-
                      🎉 no goals
                    -/
  add_left x y z :=
    Completion.induction_on₃ x y z
      (isClosed_eq
        (Continuous.inner (continuous_fst.add (continuous_fst.comp continuous_snd))
          (continuous_snd.comp continuous_snd))
        ((Continuous.inner continuous_fst (continuous_snd.comp continuous_snd)).add
          (Continuous.inner (continuous_fst.comp continuous_snd)
            (continuous_snd.comp continuous_snd))))
                      /-
                        𝕜 : Type u_1
                        E : Type u_2
                        F : Type u_3
                        inst✝² : RCLike 𝕜
                        inst✝¹ : SeminormedAddCommGroup E
                        inst✝ : InnerProductSpace 𝕜 E
                        x y z : UniformSpace.Completion E
                        a b c : E
                        ⊢ Eq (Inner.inner (HAdd.hAdd (↑E a) (↑E b)) (↑E c)) (HAdd.hAdd (Inner.inner (↑ …
                      -/
      fun a b c => by simp only [← coe_add, inner_coe, inner_add_left]
                      /-
                        🎉 no goals
                      -/
  smul_left x y c :=
    Completion.induction_on₂ x y
      (isClosed_eq (Continuous.inner (continuous_fst.const_smul c) continuous_snd)
        ((continuous_mul_left _).comp (Continuous.inner continuous_fst continuous_snd)))
                    /-
                      𝕜 : Type u_1
                      E : Type u_2
                      F : Type u_3
                      inst✝² : RCLike 𝕜
                      inst✝¹ : SeminormedAddCommGroup E
                      inst✝ : InnerProductSpace 𝕜 E
                      x y : UniformSpace.Completion E
                      c : 𝕜
                      a b : E
                      ⊢ Eq (Inner.inner (HSMul.hSMul c (↑E a)) (↑E b)) (HMul.hMul ((starRingEnd 𝕜) c …
                    -/
      fun a b => by simp only [← coe_smul c a, inner_coe, inner_smul_left]
                    /-
                      🎉 no goals
                    -/


