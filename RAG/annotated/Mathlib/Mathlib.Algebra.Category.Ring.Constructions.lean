/-- The explicit cocone with tensor products as the fibered product in `CommRingCat`. -/
def pushoutCocone : Limits.PushoutCocone
    (CommRingCat.ofHom (algebraMap R A)) (CommRingCat.ofHom (algebraMap R B)) := by
  /-
    R A B : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    ⊢ CategoryTheory.Limits.PushoutCocone (CommRingCat.ofHom (algebraMap R A)) (Co …
  -/
  fapply Limits.PushoutCocone.mk
    /-
      case W
      R A B : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      ⊢ CommRingCat
    -/
  · exact CommRingCat.of (A ⊗[R] B)
    /-
      🎉 no goals
    -/
    /-
      case inl
      R A B : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      ⊢ Quiver.Hom (CommRingCat.of A) (CommRingCat.of (TensorProduct R A B))
    -/
  · exact ofHom <| Algebra.TensorProduct.includeLeftRingHom (A := A)
    /-
      🎉 no goals
    -/
    /-
      case inr
      R A B : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      ⊢ Quiver.Hom (CommRingCat.of B) (CommRingCat.of (TensorProduct R A B))
    -/
  · exact ofHom <| Algebra.TensorProduct.includeRight.toRingHom (A := B)
    /-
      🎉 no goals
    -/
    /-
      case eq
      R A B : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (algebraMap R A))  …
    -/
  · ext r
    /-
      case eq.hf.a
      R A B : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      r : ↑(CommRingCat.of R)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (algebraMap R A)) …
    -/
    trans algebraMap R (A ⊗[R] B) r
      /-
        R A B : Type u
        inst✝⁴ : CommRing R
        inst✝³ : CommRing A
        inst✝² : CommRing B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        r : ↑(CommRingCat.of R)
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (algebraMap R A)) …
      -/
    · exact Algebra.TensorProduct.includeLeft.commutes (R := R) r
      /-
        🎉 no goals
      -/
      /-
        R A B : Type u
        inst✝⁴ : CommRing R
        inst✝³ : CommRing A
        inst✝² : CommRing B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        r : ↑(CommRingCat.of R)
        ⊢ Eq ((algebraMap R (TensorProduct R A B)) r) ((CategoryTheory.CategoryStruct. …
      -/
    · exact (Algebra.TensorProduct.includeRight.commutes (R := R) r).symm
      /-
        🎉 no goals
      -/


@[simp]
theorem pushoutCocone_inl :
    (pushoutCocone R A B).inl = ofHom (Algebra.TensorProduct.includeLeftRingHom (A := A)) :=
  rfl


@[simp]
theorem pushoutCocone_inr :
    (pushoutCocone R A B).inr = ofHom (Algebra.TensorProduct.includeRight.toRingHom (A := B)) :=
  rfl


@[simp]
theorem pushoutCocone_pt :
    (pushoutCocone R A B).pt = CommRingCat.of (A ⊗[R] B) :=
  rfl


/-- Verify that the `pushout_cocone` is indeed the colimit. -/
def pushoutCoconeIsColimit : Limits.IsColimit (pushoutCocone R A B) :=
  Limits.PushoutCocone.isColimitAux' _ fun s => by
    /-
      R A B : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      s : CategoryTheory.Limits.PushoutCocone (CommRingCat.ofHom (algebraMap R A)) ( …
      ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.pu …
    -/
    letI := RingHom.toAlgebra (s.inl.hom.comp (algebraMap R A))
    let f' : A →ₐ[R] s.pt :=
      { s.inl.hom with
        commutes' := fun r => rfl }
    let g' : B →ₐ[R] s.pt :=
      { s.inr.hom with
        commutes' := DFunLike.congr_fun <| congrArg Hom.hom
          ((s.ι.naturality Limits.WalkingSpan.Hom.snd).trans
            (s.ι.naturality Limits.WalkingSpan.Hom.fst).symm) }
    /-
      R A B : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      s : CategoryTheory.Limits.PushoutCocone (CommRingCat.ofHom (algebraMap R A)) ( …
      this : Algebra R ↑s.pt := (s.inl.hom.comp (algebraMap R A)).toAlgebra
      f' : AlgHom R A ↑s.pt :=
        let __src := s.inl.hom;
        { toRingHom := __src, commutes' := ⋯ }
      g' : AlgHom R B ↑s.pt :=
        let __src := s.inr.hom;
        { toRingHom := __src, commutes' := ⋯ }
      ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.pu …
    -/
    letI : Algebra R (pushoutCocone R A B).pt := show Algebra R (A ⊗[R] B) by infer_instance
    -- The factor map is a ⊗ b ↦ f(a) * g(b).
    /-
      R A B : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      s : CategoryTheory.Limits.PushoutCocone (CommRingCat.ofHom (algebraMap R A)) ( …
      this✝ : Algebra R ↑s.pt := (s.inl.hom.comp (algebraMap R A)).toAlgebra
      f' : AlgHom R A ↑s.pt :=
        let __src := s.inl.hom;
        { toRingHom := __src, commutes' := ⋯ }
      g' : AlgHom R B ↑s.pt :=
        let __src := s.inr.hom;
        { toRingHom := __src, commutes' := ⋯ }
      this : Algebra R ↑(CommRingCat.pushoutCocone R A B).pt := letFun inferInstance …
      ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.pu …
    -/
    use ofHom (AlgHom.toRingHom (Algebra.TensorProduct.productMap f' g'))
    /-
      case property
      R A B : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      s : CategoryTheory.Limits.PushoutCocone (CommRingCat.ofHom (algebraMap R A)) ( …
      this✝ : Algebra R ↑s.pt := (s.inl.hom.comp (algebraMap R A)).toAlgebra
      f' : AlgHom R A ↑s.pt :=
        let __src := s.inl.hom;
        { toRingHom := __src, commutes' := ⋯ }
      g' : AlgHom R B ↑s.pt :=
        let __src := s.inr.hom;
        { toRingHom := __src, commutes' := ⋯ }
      this : Algebra R ↑(CommRingCat.pushoutCocone R A B).pt := letFun inferInstance …
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.pushoutCocone R A B …
    -/
    simp only [pushoutCocone_inl, pushoutCocone_inr]
    /-
      case property
      R A B : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      s : CategoryTheory.Limits.PushoutCocone (CommRingCat.ofHom (algebraMap R A)) ( …
      this✝ : Algebra R ↑s.pt := (s.inl.hom.comp (algebraMap R A)).toAlgebra
      f' : AlgHom R A ↑s.pt :=
        let __src := s.inl.hom;
        { toRingHom := __src, commutes' := ⋯ }
      g' : AlgHom R B ↑s.pt :=
        let __src := s.inr.hom;
        { toRingHom := __src, commutes' := ⋯ }
      this : Algebra R ↑(CommRingCat.pushoutCocone R A B).pt := letFun inferInstance …
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom Algebra.Tenso …
    -/
    constructor
      /-
        case property.left
        R A B : Type u
        inst✝⁴ : CommRing R
        inst✝³ : CommRing A
        inst✝² : CommRing B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        s : CategoryTheory.Limits.PushoutCocone (CommRingCat.ofHom (algebraMap R A)) ( …
        this✝ : Algebra R ↑s.pt := (s.inl.hom.comp (algebraMap R A)).toAlgebra
        f' : AlgHom R A ↑s.pt :=
          let __src := s.inl.hom;
          { toRingHom := __src, commutes' := ⋯ }
        g' : AlgHom R B ↑s.pt :=
          let __src := s.inr.hom;
          { toRingHom := __src, commutes' := ⋯ }
        this : Algebra R ↑(CommRingCat.pushoutCocone R A B).pt := letFun inferInstance …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom Algebra.TensorProd …
      -/
    · ext x
      /-
        case property.left.hf.a
        R A B : Type u
        inst✝⁴ : CommRing R
        inst✝³ : CommRing A
        inst✝² : CommRing B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        s : CategoryTheory.Limits.PushoutCocone (CommRingCat.ofHom (algebraMap R A)) ( …
        this✝ : Algebra R ↑s.pt := (s.inl.hom.comp (algebraMap R A)).toAlgebra
        f' : AlgHom R A ↑s.pt :=
          let __src := s.inl.hom;
          { toRingHom := __src, commutes' := ⋯ }
        g' : AlgHom R B ↑s.pt :=
          let __src := s.inr.hom;
          { toRingHom := __src, commutes' := ⋯ }
        this : Algebra R ↑(CommRingCat.pushoutCocone R A B).pt := letFun inferInstance …
        x : ↑(CommRingCat.of A)
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom Algebra.TensorPro …
      -/
      exact Algebra.TensorProduct.productMap_left_apply (A := A) _ _ x
      /-
        🎉 no goals
      -/
    /-
      case property.right
      R A B : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      s : CategoryTheory.Limits.PushoutCocone (CommRingCat.ofHom (algebraMap R A)) ( …
      this✝ : Algebra R ↑s.pt := (s.inl.hom.comp (algebraMap R A)).toAlgebra
      f' : AlgHom R A ↑s.pt :=
        let __src := s.inl.hom;
        { toRingHom := __src, commutes' := ⋯ }
      g' : AlgHom R B ↑s.pt :=
        let __src := s.inr.hom;
        { toRingHom := __src, commutes' := ⋯ }
      this : Algebra R ↑(CommRingCat.pushoutCocone R A B).pt := letFun inferInstance …
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom Algebra.Tenso …
    -/
    constructor
      /-
        case property.right.left
        R A B : Type u
        inst✝⁴ : CommRing R
        inst✝³ : CommRing A
        inst✝² : CommRing B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        s : CategoryTheory.Limits.PushoutCocone (CommRingCat.ofHom (algebraMap R A)) ( …
        this✝ : Algebra R ↑s.pt := (s.inl.hom.comp (algebraMap R A)).toAlgebra
        f' : AlgHom R A ↑s.pt :=
          let __src := s.inl.hom;
          { toRingHom := __src, commutes' := ⋯ }
        g' : AlgHom R B ↑s.pt :=
          let __src := s.inr.hom;
          { toRingHom := __src, commutes' := ⋯ }
        this : Algebra R ↑(CommRingCat.pushoutCocone R A B).pt := letFun inferInstance …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom Algebra.TensorProd …
      -/
    · ext x
      /-
        case property.right.left.hf.a
        R A B : Type u
        inst✝⁴ : CommRing R
        inst✝³ : CommRing A
        inst✝² : CommRing B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        s : CategoryTheory.Limits.PushoutCocone (CommRingCat.ofHom (algebraMap R A)) ( …
        this✝ : Algebra R ↑s.pt := (s.inl.hom.comp (algebraMap R A)).toAlgebra
        f' : AlgHom R A ↑s.pt :=
          let __src := s.inl.hom;
          { toRingHom := __src, commutes' := ⋯ }
        g' : AlgHom R B ↑s.pt :=
          let __src := s.inr.hom;
          { toRingHom := __src, commutes' := ⋯ }
        this : Algebra R ↑(CommRingCat.pushoutCocone R A B).pt := letFun inferInstance …
        x : ↑(CommRingCat.of B)
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom Algebra.TensorPro …
      -/
      exact Algebra.TensorProduct.productMap_right_apply (B := B) _ _ x
      /-
        🎉 no goals
      -/
    /-
      case property.right.right
      R A B : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      s : CategoryTheory.Limits.PushoutCocone (CommRingCat.ofHom (algebraMap R A)) ( …
      this✝ : Algebra R ↑s.pt := (s.inl.hom.comp (algebraMap R A)).toAlgebra
      f' : AlgHom R A ↑s.pt :=
        let __src := s.inl.hom;
        { toRingHom := __src, commutes' := ⋯ }
      g' : AlgHom R B ↑s.pt :=
        let __src := s.inr.hom;
        { toRingHom := __src, commutes' := ⋯ }
      this : Algebra R ↑(CommRingCat.pushoutCocone R A B).pt := letFun inferInstance …
      ⊢ ∀ {m : Quiver.Hom (CommRingCat.pushoutCocone R A B).pt s.pt}, Eq (CategoryTh …
    -/
    intro h eq1 eq2
    let h' : A ⊗[R] B →ₐ[R] s.pt :=
      { h.hom with
        commutes' := fun r => by
          change h (algebraMap R A r ⊗ₜ[R] 1) = s.inl (algebraMap R A r)
          rw [← eq1]
          simp only [pushoutCocone_pt, coe_of, AlgHom.toRingHom_eq_coe]
          rfl }
    suffices h' = Algebra.TensorProduct.productMap f' g' by
      ext x
      change h' x = Algebra.TensorProduct.productMap f' g' x
      rw [this]
    /-
      case property.right.right
      R A B : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      s : CategoryTheory.Limits.PushoutCocone (CommRingCat.ofHom (algebraMap R A)) ( …
      this✝ : Algebra R ↑s.pt := (s.inl.hom.comp (algebraMap R A)).toAlgebra
      f' : AlgHom R A ↑s.pt :=
        let __src := s.inl.hom;
        { toRingHom := __src, commutes' := ⋯ }
      g' : AlgHom R B ↑s.pt :=
        let __src := s.inr.hom;
        { toRingHom := __src, commutes' := ⋯ }
      this : Algebra R ↑(CommRingCat.pushoutCocone R A B).pt := letFun inferInstance …
      h : Quiver.Hom (CommRingCat.pushoutCocone R A B).pt s.pt
      eq1 : Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom Algebra.Tensor …
      eq2 : Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom Algebra.Tensor …
      h' : AlgHom R (TensorProduct R A B) ↑s.pt :=
        let __src := h.hom;
        { toRingHom := __src, commutes' := ⋯ }
      ⊢ Eq h' (Algebra.TensorProduct.productMap f' g')
    -/
    apply Algebra.TensorProduct.ext'
    /-
      case property.right.right.H
      R A B : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      s : CategoryTheory.Limits.PushoutCocone (CommRingCat.ofHom (algebraMap R A)) ( …
      this✝ : Algebra R ↑s.pt := (s.inl.hom.comp (algebraMap R A)).toAlgebra
      f' : AlgHom R A ↑s.pt :=
        let __src := s.inl.hom;
        { toRingHom := __src, commutes' := ⋯ }
      g' : AlgHom R B ↑s.pt :=
        let __src := s.inr.hom;
        { toRingHom := __src, commutes' := ⋯ }
      this : Algebra R ↑(CommRingCat.pushoutCocone R A B).pt := letFun inferInstance …
      h : Quiver.Hom (CommRingCat.pushoutCocone R A B).pt s.pt
      eq1 : Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom Algebra.Tensor …
      eq2 : Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom Algebra.Tensor …
      h' : AlgHom R (TensorProduct R A B) ↑s.pt :=
        let __src := h.hom;
        { toRingHom := __src, commutes' := ⋯ }
      ⊢ ∀ (a : A) (b : B), Eq (h' (TensorProduct.tmul R a b)) ((Algebra.TensorProduc …
    -/
    intro a b
    simp only [f', g', ← eq1, pushoutCocone_pt, ← eq2, AlgHom.toRingHom_eq_coe,
      Algebra.TensorProduct.productMap_apply_tmul, AlgHom.coe_mk]
    /-
      case property.right.right.H
      R A B : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      s : CategoryTheory.Limits.PushoutCocone (CommRingCat.ofHom (algebraMap R A)) ( …
      this✝ : Algebra R ↑s.pt := (s.inl.hom.comp (algebraMap R A)).toAlgebra
      f' : AlgHom R A ↑s.pt :=
        let __src := s.inl.hom;
        { toRingHom := __src, commutes' := ⋯ }
      g' : AlgHom R B ↑s.pt :=
        let __src := s.inr.hom;
        { toRingHom := __src, commutes' := ⋯ }
      this : Algebra R ↑(CommRingCat.pushoutCocone R A B).pt := letFun inferInstance …
      h : Quiver.Hom (CommRingCat.pushoutCocone R A B).pt s.pt
      eq1 : Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom Algebra.Tensor …
      eq2 : Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom Algebra.Tensor …
      h' : AlgHom R (TensorProduct R A B) ↑s.pt :=
        let __src := h.hom;
        { toRingHom := __src, commutes' := ⋯ }
      a : A
      b : B
      ⊢ Eq (h' (TensorProduct.tmul R a b)) (HMul.hMul ((CategoryTheory.CategoryStruc …
    -/
    change _ = h (a ⊗ₜ 1) * h (1 ⊗ₜ b)
    /-
      case property.right.right.H
      R A B : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      s : CategoryTheory.Limits.PushoutCocone (CommRingCat.ofHom (algebraMap R A)) ( …
      this✝ : Algebra R ↑s.pt := (s.inl.hom.comp (algebraMap R A)).toAlgebra
      f' : AlgHom R A ↑s.pt :=
        let __src := s.inl.hom;
        { toRingHom := __src, commutes' := ⋯ }
      g' : AlgHom R B ↑s.pt :=
        let __src := s.inr.hom;
        { toRingHom := __src, commutes' := ⋯ }
      this : Algebra R ↑(CommRingCat.pushoutCocone R A B).pt := letFun inferInstance …
      h : Quiver.Hom (CommRingCat.pushoutCocone R A B).pt s.pt
      eq1 : Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom Algebra.Tensor …
      eq2 : Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom Algebra.Tensor …
      h' : AlgHom R (TensorProduct R A B) ↑s.pt :=
        let __src := h.hom;
        { toRingHom := __src, commutes' := ⋯ }
      a : A
      b : B
      ⊢ Eq (h' (TensorProduct.tmul R a b)) (HMul.hMul (h.hom (TensorProduct.tmul R a …
    -/
    rw [← h.hom.map_mul, Algebra.TensorProduct.tmul_mul_tmul, mul_one, one_mul]
    /-
      case property.right.right.H
      R A B : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      s : CategoryTheory.Limits.PushoutCocone (CommRingCat.ofHom (algebraMap R A)) ( …
      this✝ : Algebra R ↑s.pt := (s.inl.hom.comp (algebraMap R A)).toAlgebra
      f' : AlgHom R A ↑s.pt :=
        let __src := s.inl.hom;
        { toRingHom := __src, commutes' := ⋯ }
      g' : AlgHom R B ↑s.pt :=
        let __src := s.inr.hom;
        { toRingHom := __src, commutes' := ⋯ }
      this : Algebra R ↑(CommRingCat.pushoutCocone R A B).pt := letFun inferInstance …
      h : Quiver.Hom (CommRingCat.pushoutCocone R A B).pt s.pt
      eq1 : Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom Algebra.Tensor …
      eq2 : Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom Algebra.Tensor …
      h' : AlgHom R (TensorProduct R A B) ↑s.pt :=
        let __src := h.hom;
        { toRingHom := __src, commutes' := ⋯ }
      a : A
      b : B
      ⊢ Eq (h' (TensorProduct.tmul R a b)) (h.hom (TensorProduct.tmul R a b))
    -/
    rfl
    /-
      🎉 no goals
    -/


lemma isPushout_tensorProduct (R A B : Type u) [CommRing R] [CommRing A] [CommRing B]
    [Algebra R A] [Algebra R B] :
    IsPushout (ofHom <| algebraMap R A) (ofHom <| algebraMap R B)
      (ofHom (S := A ⊗[R] B) <| Algebra.TensorProduct.includeLeftRingHom)
      (ofHom (S := A ⊗[R] B) <| Algebra.TensorProduct.includeRight.toRingHom) where
  w := by
    /-
      R A B : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (algebraMap R A))  …
    -/
    ext
    /-
      case hf.a
      R A B : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      x✝ : ↑(CommRingCat.of R)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (algebraMap R A)) …
    -/
    simp
    /-
      🎉 no goals
    -/
  isColimit' := ⟨pushoutCoconeIsColimit R A B⟩


/-- The tensor product `A ⊗[ℤ] B` forms a cocone for `A` and `B`. -/
@[simps! pt ι]
def coproductCocone : BinaryCofan A B :=
  BinaryCofan.mk
    (ofHom (Algebra.TensorProduct.includeLeft (S := ℤ)).toRingHom : A ⟶  of (A ⊗[ℤ] B))
    (ofHom (Algebra.TensorProduct.includeRight (R := ℤ)).toRingHom : B ⟶  of (A ⊗[ℤ] B))


@[simp]
theorem coproductCocone_inl : (coproductCocone A B).inl =
  ofHom (Algebra.TensorProduct.includeLeft (S := ℤ)).toRingHom := rfl


@[simp]
theorem coproductCocone_inr : (coproductCocone A B).inr =
  ofHom (Algebra.TensorProduct.includeRight (R := ℤ)).toRingHom := rfl


/-- The tensor product `A ⊗[ℤ] B` is a coproduct for `A` and `B`. -/
@[simps]
def coproductCoconeIsColimit : IsColimit (coproductCocone A B) where
  desc (s : BinaryCofan A B) :=
    ofHom (Algebra.TensorProduct.lift s.inl.hom.toIntAlgHom s.inr.hom.toIntAlgHom
                     /-
                       A B : CommRingCat
                       s : CategoryTheory.Limits.BinaryCofan A B
                       x✝¹ : ↑((CategoryTheory.Limits.pair A B).obj { as := CategoryTheory.Limits.Wal …
                       x✝ : ↑((CategoryTheory.Limits.pair A B).obj { as := CategoryTheory.Limits.Walk …
                       ⊢ Commute (s.inl.hom.toIntAlgHom x✝¹) (s.inr.hom.toIntAlgHom x✝)
                     -/
      (fun _ _ => by apply Commute.all)).toRingHom
                     /-
                       🎉 no goals
                     -/
                                             /-
                                               A B : CommRingCat
                                               s : CategoryTheory.Limits.BinaryCofan A B
                                               x✝ : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
                                               j : CategoryTheory.Limits.WalkingPair
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp ((A.coproductCocone B).ι.app { as :=  …
                                             -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  fac (s : BinaryCofan A B) := fun ⟨j⟩ => by cases j <;> ext a <;> simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  uniq (s : BinaryCofan A B) := by
    /-
      A B : CommRingCat
      s : CategoryTheory.Limits.BinaryCofan A B
      ⊢ ∀ (m : Quiver.Hom (A.coproductCocone B).pt s.pt), (∀ (j : CategoryTheory.Dis …
    -/
    rintro ⟨m : A ⊗[ℤ] B →+* s.pt⟩ hm
    /-
      case _private.Mathlib.Algebra.Category.Ring.Basic.0.CommRingCat.Hom.mk
      A B : CommRingCat
      s : CategoryTheory.Limits.BinaryCofan A B
      m : RingHom (TensorProduct Int ↑A ↑B) ↑s.pt
      hm : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
      ⊢ Eq { hom := m } ((fun s => CommRingCat.ofHom (Algebra.TensorProduct.lift s.i …
    -/
    apply CommRingCat.hom_ext
    /-
      case _private.Mathlib.Algebra.Category.Ring.Basic.0.CommRingCat.Hom.mk.hf
      A B : CommRingCat
      s : CategoryTheory.Limits.BinaryCofan A B
      m : RingHom (TensorProduct Int ↑A ↑B) ↑s.pt
      hm : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
      ⊢ Eq { hom := m }.hom ((fun s => CommRingCat.ofHom (Algebra.TensorProduct.lift …
    -/
    apply RingHom.toIntAlgHom_injective
    /-
      case _private.Mathlib.Algebra.Category.Ring.Basic.0.CommRingCat.Hom.mk.hf.a
      A B : CommRingCat
      s : CategoryTheory.Limits.BinaryCofan A B
      m : RingHom (TensorProduct Int ↑A ↑B) ↑s.pt
      hm : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
      ⊢ Eq { hom := m }.hom.toIntAlgHom ((fun s => CommRingCat.ofHom (Algebra.Tensor …
    -/
    apply Algebra.TensorProduct.liftEquiv.symm.injective
    /-
      case _private.Mathlib.Algebra.Category.Ring.Basic.0.CommRingCat.Hom.mk.hf.a.a
      A B : CommRingCat
      s : CategoryTheory.Limits.BinaryCofan A B
      m : RingHom (TensorProduct Int ↑A ↑B) ↑s.pt
      hm : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
      ⊢ Eq (Algebra.TensorProduct.liftEquiv.symm { hom := m }.hom.toIntAlgHom) (Alge …
    -/
    apply Subtype.ext
    /-
      case _private.Mathlib.Algebra.Category.Ring.Basic.0.CommRingCat.Hom.mk.hf.a.a.a
      A B : CommRingCat
      s : CategoryTheory.Limits.BinaryCofan A B
      m : RingHom (TensorProduct Int ↑A ↑B) ↑s.pt
      hm : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
      ⊢ Eq ↑(Algebra.TensorProduct.liftEquiv.symm { hom := m }.hom.toIntAlgHom) ↑(Al …
    -/
    rw [Algebra.TensorProduct.liftEquiv_symm_apply_coe, Prod.mk.injEq]
    /-
      case _private.Mathlib.Algebra.Category.Ring.Basic.0.CommRingCat.Hom.mk.hf.a.a.a
      A B : CommRingCat
      s : CategoryTheory.Limits.BinaryCofan A B
      m : RingHom (TensorProduct Int ↑A ↑B) ↑s.pt
      hm : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
      ⊢ And (Eq ({ hom := m }.hom.toIntAlgHom.comp Algebra.TensorProduct.includeLeft …
    -/
    constructor
      /-
        case _private.Mathlib.Algebra.Category.Ring.Basic.0.CommRingCat.Hom.mk.hf.a.a. …
        A B : CommRingCat
        s : CategoryTheory.Limits.BinaryCofan A B
        m : RingHom (TensorProduct Int ↑A ↑B) ↑s.pt
        hm : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
        ⊢ Eq ({ hom := m }.hom.toIntAlgHom.comp Algebra.TensorProduct.includeLeft) (Al …
      -/
    · ext a
      /-
        case _private.Mathlib.Algebra.Category.Ring.Basic.0.CommRingCat.Hom.mk.hf.a.a. …
        A B : CommRingCat
        s : CategoryTheory.Limits.BinaryCofan A B
        m : RingHom (TensorProduct Int ↑A ↑B) ↑s.pt
        hm : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
        a : ↑A
        ⊢ Eq (({ hom := m }.hom.toIntAlgHom.comp Algebra.TensorProduct.includeLeft) a) …
      -/
      simp [map_one, mul_one, ←hm (Discrete.mk WalkingPair.left)]
      /-
        🎉 no goals
      -/
      /-
        case _private.Mathlib.Algebra.Category.Ring.Basic.0.CommRingCat.Hom.mk.hf.a.a. …
        A B : CommRingCat
        s : CategoryTheory.Limits.BinaryCofan A B
        m : RingHom (TensorProduct Int ↑A ↑B) ↑s.pt
        hm : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
        ⊢ Eq ((AlgHom.restrictScalars Int { hom := m }.hom.toIntAlgHom).comp Algebra.T …
      -/
    · ext b
      /-
        case _private.Mathlib.Algebra.Category.Ring.Basic.0.CommRingCat.Hom.mk.hf.a.a. …
        A B : CommRingCat
        s : CategoryTheory.Limits.BinaryCofan A B
        m : RingHom (TensorProduct Int ↑A ↑B) ↑s.pt
        hm : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
        b : ↑B
        ⊢ Eq (((AlgHom.restrictScalars Int { hom := m }.hom.toIntAlgHom).comp Algebra. …
      -/
      simp [map_one, mul_one, ←hm (Discrete.mk WalkingPair.right)]
      /-
        🎉 no goals
      -/


/-- The limit cone of the tensor product `A ⊗[ℤ] B` in `CommRingCat`. -/
def coproductColimitCocone : Limits.ColimitCocone (pair A B) :=
  ⟨_, coproductCoconeIsColimit A B⟩


instance (X : CommRingCat.{u}) : Unique (X ⟶ CommRingCat.of.{u} PUnit) :=
                         /-
                           X : CommRingCat
                           ⊢ ∀ (x y : ↑X), Eq ((↑1).toFun (HAdd.hAdd x y)) (HAdd.hAdd ((↑1).toFun x) ((↑1 …
                         -/
                         /-
                           🎉 no goals
                         -/
  ⟨⟨ofHom <| ⟨1, rfl, by simp⟩⟩, fun f ↦ by ext⟩
                                            /-
                                              🎉 no goals
                                            -/


/-- The trivial ring is the (strict) terminal object of `CommRingCat`. -/
def punitIsTerminal : IsTerminal (CommRingCat.of.{u} PUnit) :=
  IsTerminal.ofUnique _


instance commRingCat_hasStrictTerminalObjects : HasStrictTerminalObjects CommRingCat.{u} := by
  /-
    ⊢ CategoryTheory.Limits.HasStrictTerminalObjects CommRingCat
  -/
  apply hasStrictTerminalObjects_of_terminal_is_strict (CommRingCat.of PUnit)
  /-
    ⊢ ∀ (A : CommRingCat) (f : Quiver.Hom (CommRingCat.of PUnit.{u + 1}) A), Categ …
  -/
  intro X f
  /-
    X : CommRingCat
    f : Quiver.Hom (CommRingCat.of PUnit.{u + 1}) X
    ⊢ CategoryTheory.IsIso f
  -/
  refine ⟨ofHom ⟨1, rfl, by simp⟩, ?_, ?_⟩
    /-
      case refine_1
      X : CommRingCat
      f : Quiver.Hom (CommRingCat.of PUnit.{u + 1}) X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CommRingCat.ofHom { toMonoidHom := …
    -/
  · ext
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : CommRingCat
      f : Quiver.Hom (CommRingCat.of PUnit.{u + 1}) X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom { toMonoidHom := 1 …
    -/
  · ext x
    have e : (0 : X) = 1 := by
      rw [← f.hom.map_one, ← f.hom.map_zero]
    /-
      case refine_2.hf.a
      X : CommRingCat
      f : Quiver.Hom (CommRingCat.of PUnit.{u + 1}) X
      x : ↑X
      e : Eq 0 1
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom { toMonoidHom :=  …
    -/
    replace e : 0 * x = 1 * x := congr_arg (· * x) e
    /-
      case refine_2.hf.a
      X : CommRingCat
      f : Quiver.Hom (CommRingCat.of PUnit.{u + 1}) X
      x : ↑X
      e : Eq (HMul.hMul 0 x) (HMul.hMul 1 x)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom { toMonoidHom :=  …
    -/
    rw [one_mul, zero_mul, ← f.hom.map_zero] at e
    /-
      case refine_2.hf.a
      X : CommRingCat
      f : Quiver.Hom (CommRingCat.of PUnit.{u + 1}) X
      x : ↑X
      e : Eq (f.hom 0) x
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom { toMonoidHom :=  …
    -/
    exact e
    /-
      🎉 no goals
    -/


theorem subsingleton_of_isTerminal {X : CommRingCat} (hX : IsTerminal X) : Subsingleton X :=
  (hX.uniqueUpToIso punitIsTerminal).commRingCatIsoToRingEquiv.toEquiv.subsingleton_congr.mpr
                                /-
                                  X : CommRingCat
                                  hX : CategoryTheory.Limits.IsTerminal X
                                  ⊢ Subsingleton PUnit.{u_1 + 1}
                                -/
    (show Subsingleton PUnit by infer_instance)
                                /-
                                  🎉 no goals
                                -/


/-- `ℤ` is the initial object of `CommRingCat`. -/
def zIsInitial : IsInitial (CommRingCat.of ℤ) :=
  IsInitial.ofUnique (h := fun R => ⟨⟨ofHom <| Int.castRingHom R⟩,
    fun a => hom_ext <| a.hom.ext_int _⟩)


/-- `ULift.{u} ℤ` is initial in `CommRingCat`. -/
def isInitial : IsInitial (CommRingCat.of (ULift.{u} ℤ)) :=
  IsInitial.ofUnique (h := fun R ↦ ⟨⟨ofHom <| (Int.castRingHom R).comp ULift.ringEquiv.toRingHom⟩,
    fun _ ↦ by
      /-
        R : CommRingCat
        x✝ : Quiver.Hom (CommRingCat.of (ULift.{u, 0} Int)) R
        ⊢ Eq x✝ Inhabited.default
      -/
      ext : 1
      rw [← RingHom.cancel_right (f := (ULift.ringEquiv.{0, u} (α := ℤ)).symm.toRingHom)
        (hf := ULift.ringEquiv.symm.surjective)]
      /-
        case hf
        R : CommRingCat
        x✝ : Quiver.Hom (CommRingCat.of (ULift.{u, 0} Int)) R
        ⊢ Eq (x✝.hom.comp ULift.ringEquiv.symm.toRingHom) (Inhabited.default.hom.comp  …
      -/
      apply RingHom.ext_int⟩)
      /-
        🎉 no goals
      -/


/-- The product in `CommRingCat` is the cartesian product. This is the binary fan. -/
@[simps! pt]
def prodFan : BinaryFan A B :=
  BinaryFan.mk (CommRingCat.ofHom <| RingHom.fst A B) (CommRingCat.ofHom <| RingHom.snd A B)


/-- The product in `CommRingCat` is the cartesian product. -/
def prodFanIsLimit : IsLimit (prodFan A B) where
  lift c := ofHom <| RingHom.prod (c.π.app ⟨WalkingPair.left⟩).hom (c.π.app ⟨WalkingPair.right⟩).hom
  fac c j := by
    /-
      A B : CommRingCat
      c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair A B)
      j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun c => CommRingCat.ofHom ((c.π.ap …
    -/
    ext
    /-
      case hf.a
      A B : CommRingCat
      c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair A B)
      j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
      x✝ : ↑c.pt
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((fun c => CommRingCat.ofHom ((c.π.a …
    -/
    rcases j with ⟨⟨⟩⟩ <;>
    simp only [pair_obj_left, prodFan_pt, BinaryFan.π_app_left, BinaryFan.π_app_right,
      FunctorToTypes.map_comp_apply, forget_map, coe_of, RingHom.prod_apply] <;>
    /-
      case hf.a.mk.left
      A B : CommRingCat
      c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair A B)
      x✝ : ↑c.pt
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom ((CategoryTheory. …
    -/
    /-
      🎉 no goals
    -/
    rfl
    /-
      🎉 no goals
    -/
  uniq s m h := by
    /-
      A B : CommRingCat
      s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair A B)
      m : Quiver.Hom s.pt (A.prodFan B).pt
      h : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      ⊢ Eq m ((fun c => CommRingCat.ofHom ((c.π.app { as := CategoryTheory.Limits.Wa …
    -/
    ext x
    /-
      case hf.a
      A B : CommRingCat
      s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair A B)
      m : Quiver.Hom s.pt (A.prodFan B).pt
      h : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      x : ↑s.pt
      ⊢ Eq (m.hom x) (((fun c => CommRingCat.ofHom ((c.π.app { as := CategoryTheory. …
    -/
    change m x = (BinaryFan.fst s x, BinaryFan.snd s x)
    have eq1 : (m ≫ (A.prodFan B).fst) x = (BinaryFan.fst s) x :=
      congr_hom (h ⟨WalkingPair.left⟩) x
    have eq2 : (m ≫ (A.prodFan B).snd) x = (BinaryFan.snd s) x :=
      congr_hom (h ⟨WalkingPair.right⟩) x
    /-
      case hf.a
      A B : CommRingCat
      s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair A B)
      m : Quiver.Hom s.pt (A.prodFan B).pt
      h : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      x : ↑s.pt
      eq1 : Eq ((CategoryTheory.CategoryStruct.comp m (A.prodFan B).fst).hom x) ((Ca …
      eq2 : Eq ((CategoryTheory.CategoryStruct.comp m (A.prodFan B).snd).hom x) ((Ca …
      ⊢ Eq (m.hom x) { fst := (CategoryTheory.Limits.BinaryFan.fst s).hom x, snd :=  …
    -/
    rw [← eq1, ← eq2]
    /-
      case hf.a
      A B : CommRingCat
      s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair A B)
      m : Quiver.Hom s.pt (A.prodFan B).pt
      h : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      x : ↑s.pt
      eq1 : Eq ((CategoryTheory.CategoryStruct.comp m (A.prodFan B).fst).hom x) ((Ca …
      eq2 : Eq ((CategoryTheory.CategoryStruct.comp m (A.prodFan B).snd).hom x) ((Ca …
      ⊢ Eq (m.hom x) { fst := (CategoryTheory.CategoryStruct.comp m (A.prodFan B).fs …
    -/
    simp [prodFan]
    /-
      🎉 no goals
    -/


/--
The categorical product of rings is the cartesian product of rings. This is its `Fan`.
-/
@[simps! pt]
def piFan : Fan R :=
  Fan.mk (CommRingCat.of ((i : ι) → R i)) (fun i ↦ ofHom <| Pi.evalRingHom _ i)


/--
The categorical product of rings is the cartesian product of rings.
-/
def piFanIsLimit : IsLimit (piFan R) where
  lift s := ofHom <| Pi.ringHom fun i ↦ (s.π.1 ⟨i⟩).hom
                /-
                  ι : Type u
                  R : ι → CommRingCat
                  s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor R)
                  i : CategoryTheory.Discrete ι
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CommRingCat.ofHom (Pi.ring …
                -/
  fac s i := by rfl
                /-
                  🎉 no goals
                -/
  uniq _ _ h := hom_ext <| DFunLike.ext _ _ fun x ↦ funext fun i ↦
    DFunLike.congr_fun (congrArg Hom.hom <| h ⟨i⟩) x


/--
The categorical product and the usual product agrees
-/
def piIsoPi : ∏ᶜ R ≅ CommRingCat.of ((i : ι) → R i) :=
  limit.isoLimitCone ⟨_, piFanIsLimit R⟩


/--
The categorical product and the usual product agrees
-/
def _root_.RingEquiv.piEquivPi (R : ι → Type u) [∀ i, CommRing (R i)] :
    (∏ᶜ (fun i : ι ↦ CommRingCat.of (R i)) : CommRingCat.{u}) ≃+* ((i : ι) → R i) :=
  (piIsoPi (CommRingCat.of <| R ·)).commRingCatIsoToRingEquiv


/-- The equalizer in `CommRingCat` is the equalizer as sets. This is the equalizer fork. -/
def equalizerFork : Fork f g :=
  Fork.ofι (CommRingCat.ofHom (RingHom.eqLocus f.hom g.hom).subtype) <| by
      /-
        A B : CommRingCat
        f g : Quiver.Hom A B
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (f.hom.eqLocus g.h …
      -/
      ext ⟨x, e⟩
      /-
        case hf.a.mk
        A B : CommRingCat
        f g : Quiver.Hom A B
        x : ↑A
        e : Membership.mem (f.hom.eqLocus g.hom) x
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (f.hom.eqLocus g. …
      -/
      simpa using e
      /-
        🎉 no goals
      -/


/-- The equalizer in `CommRingCat` is the equalizer as sets. -/
def equalizerForkIsLimit : IsLimit (equalizerFork f g) := by
  /-
    A B : CommRingCat
    f g : Quiver.Hom A B
    ⊢ CategoryTheory.Limits.IsLimit (CommRingCat.equalizerFork f g)
  -/
  fapply Fork.IsLimit.mk'
  /-
    case create
    A B : CommRingCat
    f g : Quiver.Hom A B
    ⊢ (s : CategoryTheory.Limits.Fork f g) → Subtype fun l => And (Eq (CategoryThe …
  -/
  intro s
  /-
    case create
    A B : CommRingCat
    f g : Quiver.Hom A B
    s : CategoryTheory.Limits.Fork f g
    ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (CommRingCat. …
  -/
  use ofHom <| s.ι.hom.codRestrict _ fun x => (ConcreteCategory.congr_hom s.condition x : _)
  /-
    case property
    A B : CommRingCat
    f g : Quiver.Hom A B
    s : CategoryTheory.Limits.Fork f g
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (s.ι.hom.codR …
  -/
  constructor
    /-
      case property.left
      A B : CommRingCat
      f g : Quiver.Hom A B
      s : CategoryTheory.Limits.Fork f g
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (s.ι.hom.codRestri …
    -/
  · ext
    /-
      case property.left.hf.a
      A B : CommRingCat
      f g : Quiver.Hom A B
      s : CategoryTheory.Limits.Fork f g
      x✝ : ↑(((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingParallelPai …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (s.ι.hom.codRestr …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case property.right
      A B : CommRingCat
      f g : Quiver.Hom A B
      s : CategoryTheory.Limits.Fork f g
      ⊢ ∀ {m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.Walk …
    -/
  · intro m hm
    /-
      case property.right
      A B : CommRingCat
      f g : Quiver.Hom A B
      s : CategoryTheory.Limits.Fork f g
      m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      hm : Eq (CategoryTheory.CategoryStruct.comp m (CommRingCat.equalizerFork f g). …
      ⊢ Eq m (CommRingCat.ofHom (s.ι.hom.codRestrict (f.hom.eqLocus g.hom) ⋯))
    -/
    ext x
    /-
      case property.right.hf.a
      A B : CommRingCat
      f g : Quiver.Hom A B
      s : CategoryTheory.Limits.Fork f g
      m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      hm : Eq (CategoryTheory.CategoryStruct.comp m (CommRingCat.equalizerFork f g). …
      x : ↑(((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingParallelPair …
      ⊢ Eq (m.hom x) ((CommRingCat.ofHom (s.ι.hom.codRestrict (f.hom.eqLocus g.hom)  …
    -/
    exact Subtype.ext <| RingHom.congr_fun (congrArg Hom.hom hm) x
    /-
      🎉 no goals
    -/


instance : IsLocalHom (equalizerFork f g).ι.hom := by
  /-
    A B : CommRingCat
    f g : Quiver.Hom A B
    ⊢ IsLocalHom (CommRingCat.equalizerFork f g).ι.hom
  -/
  constructor
  /-
    case map_nonunit
    A B : CommRingCat
    f g : Quiver.Hom A B
    ⊢ ∀ (a : ↑(((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingParalle …
  -/
  rintro ⟨a, h₁ : _ = _⟩ (⟨⟨x, y, h₃, h₄⟩, rfl : x = _⟩ : IsUnit a)
  have : y ∈ RingHom.eqLocus f.hom g.hom := by
    apply (f.hom.isUnit_map ⟨⟨x, y, h₃, h₄⟩, rfl⟩ : IsUnit (f x)).mul_left_inj.mp
    conv_rhs => rw [h₁]
    rw [← f.hom.map_mul, ← g.hom.map_mul, h₄, f.hom.map_one, g.hom.map_one]
  /-
    case map_nonunit.mk.intro.mk
    A B : CommRingCat
    f g : Quiver.Hom A B
    x y : ↑A
    h₃ : Eq (HMul.hMul x y) 1
    h₄ : Eq (HMul.hMul y x) 1
    h₁ : Eq (f.hom x) (g.hom x)
    this : Membership.mem (f.hom.eqLocus g.hom) y
    ⊢ IsUnit ⟨x, h₁⟩
  -/
  rw [isUnit_iff_exists_inv]
  /-
    case map_nonunit.mk.intro.mk
    A B : CommRingCat
    f g : Quiver.Hom A B
    x y : ↑A
    h₃ : Eq (HMul.hMul x y) 1
    h₄ : Eq (HMul.hMul y x) 1
    h₁ : Eq (f.hom x) (g.hom x)
    this : Membership.mem (f.hom.eqLocus g.hom) y
    ⊢ Exists fun b => Eq (HMul.hMul ⟨x, h₁⟩ b) 1
  -/
  exact ⟨⟨y, this⟩, Subtype.eq h₃⟩
  /-
    🎉 no goals
  -/


@[instance]
theorem equalizer_ι_isLocalHom (F : WalkingParallelPair ⥤ CommRingCat.{u}) :
    IsLocalHom (limit.π F WalkingParallelPair.zero).hom := by
  /-
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair CommRingCat
    ⊢ IsLocalHom (CategoryTheory.Limits.limit.π F CategoryTheory.Limits.WalkingPar …
  -/
  have := limMap_π (diagramIsoParallelPair F).hom WalkingParallelPair.zero
  /-
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair CommRingCat
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limMap (C …
    ⊢ IsLocalHom (CategoryTheory.Limits.limit.π F CategoryTheory.Limits.WalkingPar …
  -/
  rw [← IsIso.comp_inv_eq] at this
  /-
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair CommRingCat
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
    ⊢ IsLocalHom (CategoryTheory.Limits.limit.π F CategoryTheory.Limits.WalkingPar …
  -/
  rw [← this]
  rw [← limit.isoLimitCone_hom_π
      ⟨_,
        equalizerForkIsLimit (F.map WalkingParallelPairHom.left)
          (F.map WalkingParallelPairHom.right)⟩
      WalkingParallelPair.zero]
  /-
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair CommRingCat
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
    ⊢ IsLocalHom (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruc …
  -/
  change IsLocalHom ((lim.map _ ≫ _ ≫ (equalizerFork _ _).ι) ≫ _).hom
  /-
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair CommRingCat
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
    ⊢ IsLocalHom (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruc …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-10")]
alias equalizer_ι_isLocalRingHom := equalizer_ι_isLocalHom


instance equalizer_ι_is_local_ring_hom' (F : WalkingParallelPairᵒᵖ ⥤ CommRingCat.{u}) :
    IsLocalHom (limit.π F (Opposite.op WalkingParallelPair.one)).hom := by
  have : _ = limit.π F (walkingParallelPairOpEquiv.functor.obj _) :=
    (limit.isoLimitCone_inv_π
        ⟨_, IsLimit.whiskerEquivalence (limit.isLimit F) walkingParallelPairOpEquiv⟩
        WalkingParallelPair.zero : _)
  /-
    A B : CommRingCat
    f g : Quiver.Hom A B
    F : CategoryTheory.Functor (Opposite CategoryTheory.Limits.WalkingParallelPair …
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.iso …
    ⊢ IsLocalHom (CategoryTheory.Limits.limit.π F { unop := CategoryTheory.Limits. …
  -/
  erw [← this]
  -- note: this was not needed before https://github.com/leanprover-community/mathlib4/pull/19757
  haveI : IsLocalHom (limit.π (walkingParallelPairOpEquiv.functor ⋙ F) zero).hom := by
    infer_instance
  /-
    A B : CommRingCat
    f g : Quiver.Hom A B
    F : CategoryTheory.Functor (Opposite CategoryTheory.Limits.WalkingParallelPair …
    this✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.is …
    this : IsLocalHom (CategoryTheory.Limits.limit.π (CategoryTheory.Limits.walkin …
    ⊢ IsLocalHom (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit. …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- In the category of `CommRingCat`, the pullback of `f : A ⟶ C` and `g : B ⟶ C` is the `eqLocus`
of the two maps `A × B ⟶ C`. This is the constructed pullback cone.
-/
def pullbackCone {A B C : CommRingCat.{u}} (f : A ⟶ C) (g : B ⟶ C) : PullbackCone f g :=
  PullbackCone.mk
    (CommRingCat.ofHom <|
      (RingHom.fst A B).comp
        (RingHom.eqLocus (f.hom.comp (RingHom.fst A B)) (g.hom.comp (RingHom.snd A B))).subtype)
    (CommRingCat.ofHom <|
      (RingHom.snd A B).comp
        (RingHom.eqLocus (f.hom.comp (RingHom.fst A B)) (g.hom.comp (RingHom.snd A B))).subtype)
    (by
      /-
        A B C : CommRingCat
        f : Quiver.Hom A C
        g : Quiver.Hom B C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom ((RingHom.fst ↑A ↑ …
      -/
      ext ⟨x, e⟩
      /-
        case hf.a.mk
        A B C : CommRingCat
        f : Quiver.Hom A C
        g : Quiver.Hom B C
        x : Prod ↑A ↑B
        e : Membership.mem ((f.hom.comp (RingHom.fst ↑A ↑B)).eqLocus (g.hom.comp (Ring …
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom ((RingHom.fst ↑A  …
      -/
      simpa [CommRingCat.ofHom] using e)
      /-
        🎉 no goals
      -/


/-- The constructed pullback cone is indeed the limit. -/
def pullbackConeIsLimit {A B C : CommRingCat.{u}} (f : A ⟶ C) (g : B ⟶ C) :
    IsLimit (pullbackCone f g) := by
  /-
    A B C : CommRingCat
    f : Quiver.Hom A C
    g : Quiver.Hom B C
    ⊢ CategoryTheory.Limits.IsLimit (CommRingCat.pullbackCone f g)
  -/
  fapply PullbackCone.IsLimit.mk
    /-
      case lift
      A B C : CommRingCat
      f : Quiver.Hom A C
      g : Quiver.Hom B C
      ⊢ (s : CategoryTheory.Limits.PullbackCone f g) → Quiver.Hom s.pt (CommRingCat. …
    -/
  · intro s
    /-
      case lift
      A B C : CommRingCat
      f : Quiver.Hom A C
      g : Quiver.Hom B C
      s : CategoryTheory.Limits.PullbackCone f g
      ⊢ Quiver.Hom s.pt (CommRingCat.of (Subtype fun x => Membership.mem ((f.hom.com …
    -/
    refine ofHom ((s.fst.hom.prod s.snd.hom).codRestrict _ ?_)
    /-
      case lift
      A B C : CommRingCat
      f : Quiver.Hom A C
      g : Quiver.Hom B C
      s : CategoryTheory.Limits.PullbackCone f g
      ⊢ ∀ (x : ↑s.pt), Membership.mem ((f.hom.comp (RingHom.fst ↑A ↑B)).eqLocus (g.h …
    -/
    intro x
    /-
      case lift
      A B C : CommRingCat
      f : Quiver.Hom A C
      g : Quiver.Hom B C
      s : CategoryTheory.Limits.PullbackCone f g
      x : ↑s.pt
      ⊢ Membership.mem ((f.hom.comp (RingHom.fst ↑A ↑B)).eqLocus (g.hom.comp (RingHo …
    -/
    exact congr_arg (fun f : s.pt →+* C => f x) (congrArg Hom.hom s.condition)
    /-
      🎉 no goals
    -/
    /-
      case fac_left
      A B C : CommRingCat
      f : Quiver.Hom A C
      g : Quiver.Hom B C
      ⊢ ∀ (s : CategoryTheory.Limits.PullbackCone f g), Eq (CategoryTheory.CategoryS …
    -/
  · intro s
    /-
      case fac_left
      A B C : CommRingCat
      f : Quiver.Hom A C
      g : Quiver.Hom B C
      s : CategoryTheory.Limits.PullbackCone f g
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom ((s.fst.hom.prod s …
    -/
    ext x
    /-
      case fac_left.hf.a
      A B C : CommRingCat
      f : Quiver.Hom A C
      g : Quiver.Hom B C
      s : CategoryTheory.Limits.PullbackCone f g
      x : ↑s.pt
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom ((s.fst.hom.prod  …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case fac_right
      A B C : CommRingCat
      f : Quiver.Hom A C
      g : Quiver.Hom B C
      ⊢ ∀ (s : CategoryTheory.Limits.PullbackCone f g), Eq (CategoryTheory.CategoryS …
    -/
  · intro s
    /-
      case fac_right
      A B C : CommRingCat
      f : Quiver.Hom A C
      g : Quiver.Hom B C
      s : CategoryTheory.Limits.PullbackCone f g
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom ((s.fst.hom.prod s …
    -/
    ext x
    /-
      case fac_right.hf.a
      A B C : CommRingCat
      f : Quiver.Hom A C
      g : Quiver.Hom B C
      s : CategoryTheory.Limits.PullbackCone f g
      x : ↑s.pt
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom ((s.fst.hom.prod  …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case uniq
      A B C : CommRingCat
      f : Quiver.Hom A C
      g : Quiver.Hom B C
      ⊢ ∀ (s : CategoryTheory.Limits.PullbackCone f g) (m : Quiver.Hom s.pt (CommRin …
    -/
  · intro s m e₁ e₂
    /-
      case uniq
      A B C : CommRingCat
      f : Quiver.Hom A C
      g : Quiver.Hom B C
      s : CategoryTheory.Limits.PullbackCone f g
      m : Quiver.Hom s.pt (CommRingCat.of (Subtype fun x => Membership.mem ((f.hom.c …
      e₁ : Eq (CategoryTheory.CategoryStruct.comp m (CommRingCat.ofHom ((RingHom.fst …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp m (CommRingCat.ofHom ((RingHom.snd …
      ⊢ Eq m (CommRingCat.ofHom ((s.fst.hom.prod s.snd.hom).codRestrict ((f.hom.comp …
    -/
    refine hom_ext <| RingHom.ext fun (x : s.pt) => Subtype.ext ?_
    /-
      case uniq
      A B C : CommRingCat
      f : Quiver.Hom A C
      g : Quiver.Hom B C
      s : CategoryTheory.Limits.PullbackCone f g
      m : Quiver.Hom s.pt (CommRingCat.of (Subtype fun x => Membership.mem ((f.hom.c …
      e₁ : Eq (CategoryTheory.CategoryStruct.comp m (CommRingCat.ofHom ((RingHom.fst …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp m (CommRingCat.ofHom ((RingHom.snd …
      x : ↑s.pt
      ⊢ Eq ↑(m.hom x) ↑((CommRingCat.ofHom ((s.fst.hom.prod s.snd.hom).codRestrict ( …
    -/
    change (m x).1 = (_, _)
    /-
      case uniq
      A B C : CommRingCat
      f : Quiver.Hom A C
      g : Quiver.Hom B C
      s : CategoryTheory.Limits.PullbackCone f g
      m : Quiver.Hom s.pt (CommRingCat.of (Subtype fun x => Membership.mem ((f.hom.c …
      e₁ : Eq (CategoryTheory.CategoryStruct.comp m (CommRingCat.ofHom ((RingHom.fst …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp m (CommRingCat.ofHom ((RingHom.snd …
      x : ↑s.pt
      ⊢ Eq ↑(m.hom x) { fst := s.fst.hom x, snd := s.snd.hom x }
    -/
    have eq1 := (congr_arg (fun f : s.pt →+* A => f x) (congrArg Hom.hom e₁) : _)
    /-
      case uniq
      A B C : CommRingCat
      f : Quiver.Hom A C
      g : Quiver.Hom B C
      s : CategoryTheory.Limits.PullbackCone f g
      m : Quiver.Hom s.pt (CommRingCat.of (Subtype fun x => Membership.mem ((f.hom.c …
      e₁ : Eq (CategoryTheory.CategoryStruct.comp m (CommRingCat.ofHom ((RingHom.fst …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp m (CommRingCat.ofHom ((RingHom.snd …
      x : ↑s.pt
      eq1 : Eq ((CategoryTheory.CategoryStruct.comp m (CommRingCat.ofHom ((RingHom.f …
      ⊢ Eq ↑(m.hom x) { fst := s.fst.hom x, snd := s.snd.hom x }
    -/
    have eq2 := (congr_arg (fun f : s.pt →+* B => f x) (congrArg Hom.hom e₂) : _)
    /-
      case uniq
      A B C : CommRingCat
      f : Quiver.Hom A C
      g : Quiver.Hom B C
      s : CategoryTheory.Limits.PullbackCone f g
      m : Quiver.Hom s.pt (CommRingCat.of (Subtype fun x => Membership.mem ((f.hom.c …
      e₁ : Eq (CategoryTheory.CategoryStruct.comp m (CommRingCat.ofHom ((RingHom.fst …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp m (CommRingCat.ofHom ((RingHom.snd …
      x : ↑s.pt
      eq1 : Eq ((CategoryTheory.CategoryStruct.comp m (CommRingCat.ofHom ((RingHom.f …
      eq2 : Eq ((CategoryTheory.CategoryStruct.comp m (CommRingCat.ofHom ((RingHom.s …
      ⊢ Eq ↑(m.hom x) { fst := s.fst.hom x, snd := s.snd.hom x }
    -/
    rw [← eq1, ← eq2]
    /-
      case uniq
      A B C : CommRingCat
      f : Quiver.Hom A C
      g : Quiver.Hom B C
      s : CategoryTheory.Limits.PullbackCone f g
      m : Quiver.Hom s.pt (CommRingCat.of (Subtype fun x => Membership.mem ((f.hom.c …
      e₁ : Eq (CategoryTheory.CategoryStruct.comp m (CommRingCat.ofHom ((RingHom.fst …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp m (CommRingCat.ofHom ((RingHom.snd …
      x : ↑s.pt
      eq1 : Eq ((CategoryTheory.CategoryStruct.comp m (CommRingCat.ofHom ((RingHom.f …
      eq2 : Eq ((CategoryTheory.CategoryStruct.comp m (CommRingCat.ofHom ((RingHom.s …
      ⊢ Eq ↑(m.hom x) { fst := (CategoryTheory.CategoryStruct.comp m (CommRingCat.of …
    -/
    rfl
    /-
      🎉 no goals
    -/


