theorem baseChangeAux_surj {σ : Type*} {f : MvPolynomial σ R →ₐ[R] A} (hf : Function.Surjective f) :
    Function.Surjective (Algebra.TensorProduct.map (AlgHom.id B B) f) := by
  /-
    R : Type w₁
    inst✝⁴ : CommRing R
    A : Type w₂
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    B : Type w₃
    inst✝¹ : CommRing B
    inst✝ : Algebra R B
    σ : Type u_1
    f : AlgHom R (MvPolynomial σ R) A
    hf : Function.Surjective ⇑f
    ⊢ Function.Surjective ⇑(Algebra.TensorProduct.map (AlgHom.id B B) f)
  -/
  show Function.Surjective (TensorProduct.map (AlgHom.id R B) f)
  /-
    R : Type w₁
    inst✝⁴ : CommRing R
    A : Type w₂
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    B : Type w₃
    inst✝¹ : CommRing B
    inst✝ : Algebra R B
    σ : Type u_1
    f : AlgHom R (MvPolynomial σ R) A
    hf : Function.Surjective ⇑f
    ⊢ Function.Surjective ⇑(Algebra.TensorProduct.map (AlgHom.id R B) f)
  -/
  apply TensorProduct.map_surjective
    /-
      case hg
      R : Type w₁
      inst✝⁴ : CommRing R
      A : Type w₂
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      B : Type w₃
      inst✝¹ : CommRing B
      inst✝ : Algebra R B
      σ : Type u_1
      f : AlgHom R (MvPolynomial σ R) A
      hf : Function.Surjective ⇑f
      ⊢ Function.Surjective ⇑(AlgHom.id R B).toLinearMap
    -/
  · exact Function.RightInverse.surjective (congrFun rfl)
    /-
      🎉 no goals
    -/
    /-
      case hg'
      R : Type w₁
      inst✝⁴ : CommRing R
      A : Type w₂
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      B : Type w₃
      inst✝¹ : CommRing B
      inst✝ : Algebra R B
      σ : Type u_1
      f : AlgHom R (MvPolynomial σ R) A
      hf : Function.Surjective ⇑f
      ⊢ Function.Surjective ⇑f.toLinearMap
    -/
  · exact hf
    /-
      🎉 no goals
    -/


instance baseChange [hfa : FiniteType R A] : Algebra.FiniteType B (B ⊗[R] A) := by
  /-
    R : Type w₁
    inst✝⁴ : CommRing R
    A : Type w₂
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    B : Type w₃
    inst✝¹ : CommRing B
    inst✝ : Algebra R B
    hfa : Algebra.FiniteType R A
    ⊢ Algebra.FiniteType B (TensorProduct R B A)
  -/
  rw [iff_quotient_mvPolynomial''] at *
  /-
    R : Type w₁
    inst✝⁴ : CommRing R
    A : Type w₂
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    B : Type w₃
    inst✝¹ : CommRing B
    inst✝ : Algebra R B
    hfa : Exists fun n => Exists fun f => Function.Surjective ⇑f
    ⊢ Exists fun n => Exists fun f => Function.Surjective ⇑f
  -/
  obtain ⟨n, f, hf⟩ := hfa
  let g : B ⊗[R] MvPolynomial (Fin n) R →ₐ[B] B ⊗[R] A :=
    Algebra.TensorProduct.map (AlgHom.id B B) f
  /-
    case intro.intro
    R : Type w₁
    inst✝⁴ : CommRing R
    A : Type w₂
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    B : Type w₃
    inst✝¹ : CommRing B
    inst✝ : Algebra R B
    n : Nat
    f : AlgHom R (MvPolynomial (Fin n) R) A
    hf : Function.Surjective ⇑f
    g : AlgHom B (TensorProduct R B (MvPolynomial (Fin n) R)) (TensorProduct R B A …
    ⊢ Exists fun n => Exists fun f => Function.Surjective ⇑f
  -/
  have : Function.Surjective g := baseChangeAux_surj B hf
  /-
    case intro.intro
    R : Type w₁
    inst✝⁴ : CommRing R
    A : Type w₂
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    B : Type w₃
    inst✝¹ : CommRing B
    inst✝ : Algebra R B
    n : Nat
    f : AlgHom R (MvPolynomial (Fin n) R) A
    hf : Function.Surjective ⇑f
    g : AlgHom B (TensorProduct R B (MvPolynomial (Fin n) R)) (TensorProduct R B A …
    this : Function.Surjective ⇑g
    ⊢ Exists fun n => Exists fun f => Function.Surjective ⇑f
  -/
  use n, AlgHom.comp g (MvPolynomial.algebraTensorAlgEquiv R B).symm.toAlgHom
  /-
    case h
    R : Type w₁
    inst✝⁴ : CommRing R
    A : Type w₂
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    B : Type w₃
    inst✝¹ : CommRing B
    inst✝ : Algebra R B
    n : Nat
    f : AlgHom R (MvPolynomial (Fin n) R) A
    hf : Function.Surjective ⇑f
    g : AlgHom B (TensorProduct R B (MvPolynomial (Fin n) R)) (TensorProduct R B A …
    this : Function.Surjective ⇑g
    ⊢ Function.Surjective ⇑(g.comp ↑(MvPolynomial.algebraTensorAlgEquiv R B).symm)
  -/
  simpa
  /-
    🎉 no goals
  -/


instance baseChange [FinitePresentation R A] : FinitePresentation B (B ⊗[R] A) := by
  /-
    R : Type w₁
    inst✝⁵ : CommRing R
    A : Type w₂
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    B : Type w₃
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FinitePresentation R A
    ⊢ Algebra.FinitePresentation B (TensorProduct R B A)
  -/
  obtain ⟨n, f, hsurj, hfg⟩ := ‹FinitePresentation R A›
  let g : B ⊗[R] MvPolynomial (Fin n) R →ₐ[B] B ⊗[R] A :=
    Algebra.TensorProduct.map (AlgHom.id B B) f
  /-
    case mk.intro.intro.intro
    R : Type w₁
    inst✝⁵ : CommRing R
    A : Type w₂
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    B : Type w₃
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FinitePresentation R A
    n : Nat
    f : AlgHom R (MvPolynomial (Fin n) R) A
    hsurj : Function.Surjective ⇑f
    hfg : (RingHom.ker f.toRingHom).FG
    g : AlgHom B (TensorProduct R B (MvPolynomial (Fin n) R)) (TensorProduct R B A …
    ⊢ Algebra.FinitePresentation B (TensorProduct R B A)
  -/
  have hgsurj : Function.Surjective g := Algebra.FiniteType.baseChangeAux_surj B hsurj
  have hker_eq : RingHom.ker g = Ideal.map Algebra.TensorProduct.includeRight (RingHom.ker f) :=
    Algebra.TensorProduct.lTensor_ker f hsurj
  have hfgg : Ideal.FG (RingHom.ker g) := by
    rw [hker_eq]
    exact Ideal.FG.map hfg _
  let g' : MvPolynomial (Fin n) B →ₐ[B] B ⊗[R] A :=
    AlgHom.comp g (MvPolynomial.algebraTensorAlgEquiv R B).symm.toAlgHom
  /-
    case mk.intro.intro.intro
    R : Type w₁
    inst✝⁵ : CommRing R
    A : Type w₂
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    B : Type w₃
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FinitePresentation R A
    n : Nat
    f : AlgHom R (MvPolynomial (Fin n) R) A
    hsurj : Function.Surjective ⇑f
    hfg : (RingHom.ker f.toRingHom).FG
    g : AlgHom B (TensorProduct R B (MvPolynomial (Fin n) R)) (TensorProduct R B A …
    hgsurj : Function.Surjective ⇑g
    hker_eq : Eq (RingHom.ker g) (Ideal.map Algebra.TensorProduct.includeRight (Ri …
    hfgg : (RingHom.ker g).FG
    g' : AlgHom B (MvPolynomial (Fin n) B) (TensorProduct R B A) := g.comp ↑(MvPol …
    ⊢ Algebra.FinitePresentation B (TensorProduct R B A)
  -/
  refine ⟨n, g', ?_, Ideal.fg_ker_comp _ _ ?_ hfgg ?_⟩
    /-
      case mk.intro.intro.intro.refine_1
      R : Type w₁
      inst✝⁵ : CommRing R
      A : Type w₂
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      B : Type w₃
      inst✝² : CommRing B
      inst✝¹ : Algebra R B
      inst✝ : Algebra.FinitePresentation R A
      n : Nat
      f : AlgHom R (MvPolynomial (Fin n) R) A
      hsurj : Function.Surjective ⇑f
      hfg : (RingHom.ker f.toRingHom).FG
      g : AlgHom B (TensorProduct R B (MvPolynomial (Fin n) R)) (TensorProduct R B A …
      hgsurj : Function.Surjective ⇑g
      hker_eq : Eq (RingHom.ker g) (Ideal.map Algebra.TensorProduct.includeRight (Ri …
      hfgg : (RingHom.ker g).FG
      g' : AlgHom B (MvPolynomial (Fin n) B) (TensorProduct R B A) := g.comp ↑(MvPol …
      ⊢ Function.Surjective ⇑g'
    -/
  · simp_all [g, g']
    /-
      🎉 no goals
    -/
    /-
      case mk.intro.intro.intro.refine_2
      R : Type w₁
      inst✝⁵ : CommRing R
      A : Type w₂
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      B : Type w₃
      inst✝² : CommRing B
      inst✝¹ : Algebra R B
      inst✝ : Algebra.FinitePresentation R A
      n : Nat
      f : AlgHom R (MvPolynomial (Fin n) R) A
      hsurj : Function.Surjective ⇑f
      hfg : (RingHom.ker f.toRingHom).FG
      g : AlgHom B (TensorProduct R B (MvPolynomial (Fin n) R)) (TensorProduct R B A …
      hgsurj : Function.Surjective ⇑g
      hker_eq : Eq (RingHom.ker g) (Ideal.map Algebra.TensorProduct.includeRight (Ri …
      hfgg : (RingHom.ker g).FG
      g' : AlgHom B (MvPolynomial (Fin n) B) (TensorProduct R B A) := g.comp ↑(MvPol …
      ⊢ (RingHom.ker ↑↑(MvPolynomial.algebraTensorAlgEquiv R B).symm).FG
    -/
  · show Ideal.FG (RingHom.ker (AlgEquiv.symm (MvPolynomial.algebraTensorAlgEquiv R B)))
    /-
      case mk.intro.intro.intro.refine_2
      R : Type w₁
      inst✝⁵ : CommRing R
      A : Type w₂
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      B : Type w₃
      inst✝² : CommRing B
      inst✝¹ : Algebra R B
      inst✝ : Algebra.FinitePresentation R A
      n : Nat
      f : AlgHom R (MvPolynomial (Fin n) R) A
      hsurj : Function.Surjective ⇑f
      hfg : (RingHom.ker f.toRingHom).FG
      g : AlgHom B (TensorProduct R B (MvPolynomial (Fin n) R)) (TensorProduct R B A …
      hgsurj : Function.Surjective ⇑g
      hker_eq : Eq (RingHom.ker g) (Ideal.map Algebra.TensorProduct.includeRight (Ri …
      hfgg : (RingHom.ker g).FG
      g' : AlgHom B (MvPolynomial (Fin n) B) (TensorProduct R B A) := g.comp ↑(MvPol …
      ⊢ (RingHom.ker (MvPolynomial.algebraTensorAlgEquiv R B).symm).FG
    -/
    simp only [RingHom.ker_equiv]
    /-
      case mk.intro.intro.intro.refine_2
      R : Type w₁
      inst✝⁵ : CommRing R
      A : Type w₂
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      B : Type w₃
      inst✝² : CommRing B
      inst✝¹ : Algebra R B
      inst✝ : Algebra.FinitePresentation R A
      n : Nat
      f : AlgHom R (MvPolynomial (Fin n) R) A
      hsurj : Function.Surjective ⇑f
      hfg : (RingHom.ker f.toRingHom).FG
      g : AlgHom B (TensorProduct R B (MvPolynomial (Fin n) R)) (TensorProduct R B A …
      hgsurj : Function.Surjective ⇑g
      hker_eq : Eq (RingHom.ker g) (Ideal.map Algebra.TensorProduct.includeRight (Ri …
      hfgg : (RingHom.ker g).FG
      g' : AlgHom B (MvPolynomial (Fin n) B) (TensorProduct R B A) := g.comp ↑(MvPol …
      ⊢ Bot.bot.FG
    -/
    exact Submodule.fg_bot
    /-
      🎉 no goals
    -/
    /-
      case mk.intro.intro.intro.refine_3
      R : Type w₁
      inst✝⁵ : CommRing R
      A : Type w₂
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      B : Type w₃
      inst✝² : CommRing B
      inst✝¹ : Algebra R B
      inst✝ : Algebra.FinitePresentation R A
      n : Nat
      f : AlgHom R (MvPolynomial (Fin n) R) A
      hsurj : Function.Surjective ⇑f
      hfg : (RingHom.ker f.toRingHom).FG
      g : AlgHom B (TensorProduct R B (MvPolynomial (Fin n) R)) (TensorProduct R B A …
      hgsurj : Function.Surjective ⇑g
      hker_eq : Eq (RingHom.ker g) (Ideal.map Algebra.TensorProduct.includeRight (Ri …
      hfgg : (RingHom.ker g).FG
      g' : AlgHom B (MvPolynomial (Fin n) B) (TensorProduct R B A) := g.comp ↑(MvPol …
      ⊢ Function.Surjective ⇑↑↑(MvPolynomial.algebraTensorAlgEquiv R B).symm
    -/
  · simpa using EquivLike.surjective _
    /-
      🎉 no goals
    -/


