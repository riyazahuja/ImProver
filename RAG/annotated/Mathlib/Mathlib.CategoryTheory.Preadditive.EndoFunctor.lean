/-- The category of algebras over an additive endofunctor on a preadditive category is preadditive.
-/
@[simps]
instance Endofunctor.algebraPreadditive : Preadditive (Endofunctor.Algebra F) where
  homGroup A₁ A₂ :=
    { add := fun α β =>
        { f := α.f + β.f
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    inst✝¹ : CategoryTheory.Preadditive C
                    F : CategoryTheory.Functor C C
                    inst✝ : F.Additive
                    A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
                    α β : Quiver.Hom A₁ A₂
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (HAdd.hAdd α.f β.f)) A₂.str) ( …
                  -/
          h := by simp only [Functor.map_add, add_comp, Endofunctor.Algebra.Hom.h, comp_add] }
                  /-
                    🎉 no goals
                  -/
      zero :=
        { f := 0
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    inst✝¹ : CategoryTheory.Preadditive C
                    F : CategoryTheory.Functor C C
                    inst✝ : F.Additive
                    A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map 0) A₂.str) (CategoryTheory.Cat …
                  -/
          h := by simp only [Functor.map_zero, zero_comp, comp_zero] }
                  /-
                    🎉 no goals
                  -/
      nsmul := fun n α =>
        { f := n • α.f
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    inst✝¹ : CategoryTheory.Preadditive C
                    F : CategoryTheory.Functor C C
                    inst✝ : F.Additive
                    A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
                    n : Nat
                    α : Quiver.Hom A₁ A₂
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (HSMul.hSMul n α.f)) A₂.str) ( …
                  -/
          h := by rw [comp_nsmul, Functor.map_nsmul, nsmul_comp, Endofunctor.Algebra.Hom.h] }
                  /-
                    🎉 no goals
                  -/
      neg := fun α =>
        { f := -α.f
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    inst✝¹ : CategoryTheory.Preadditive C
                    F : CategoryTheory.Functor C C
                    inst✝ : F.Additive
                    A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
                    α : Quiver.Hom A₁ A₂
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (Neg.neg α.f)) A₂.str) (Catego …
                  -/
          h := by simp only [Functor.map_neg, neg_comp, Endofunctor.Algebra.Hom.h, comp_neg] }
                  /-
                    🎉 no goals
                  -/
      sub := fun α β =>
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          ⊢ ∀ (a b c : Quiver.Hom A₁ A₂), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a  …
        -/
        { f := α.f - β.f
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          a✝ b✝ c✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd (HAdd.hAdd a✝ b✝) c✝) (HAdd.hAdd a✝ (HAdd.hAdd b✝ c✝))
        -/
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    inst✝¹ : CategoryTheory.Preadditive C
                    F : CategoryTheory.Functor C C
                    inst✝ : F.Additive
                    A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
                    α β : Quiver.Hom A₁ A₂
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (HSub.hSub α.f β.f)) A₂.str) ( …
                  -/
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          a✝ b✝ c✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd (HAdd.hAdd a✝ b✝) c✝).f (HAdd.hAdd a✝ (HAdd.hAdd b✝ c✝)).f
        -/
          h := by simp only [Functor.map_sub, sub_comp, Endofunctor.Algebra.Hom.h, comp_sub] }
        /-
          🎉 no goals
        -/
                  /-
                    🎉 no goals
                  -/
      zsmul := fun r α =>
        { f := r • α.f
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          ⊢ ∀ (a : Quiver.Hom A₁ A₂), Eq (HAdd.hAdd 0 a) a
        -/
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    inst✝¹ : CategoryTheory.Preadditive C
                    F : CategoryTheory.Functor C C
                    inst✝ : F.Additive
                    A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
                    r : Int
                    α : Quiver.Hom A₁ A₂
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (HSMul.hSMul r α.f)) A₂.str) ( …
                  -/
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd 0 a✝) a✝
        -/
          h := by rw [comp_zsmul, Functor.map_zsmul, zsmul_comp, Endofunctor.Algebra.Hom.h] }
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd 0 a✝).f a✝.f
        -/
                  /-
                    🎉 no goals
                  -/
        /-
          🎉 no goals
        -/
      add_assoc := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          ⊢ ∀ (a : Quiver.Hom A₁ A₂), Eq (HAdd.hAdd a 0) a
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd a✝ 0) a✝
        -/
        apply Algebra.Hom.ext
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd a✝ 0).f a✝.f
        -/
        apply add_assoc
        /-
          🎉 no goals
        -/
      zero_add := by
        intros
        apply Algebra.Hom.ext
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          ⊢ ∀ (x : Quiver.Hom A₁ A₂), Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ } …
        -/
        apply zero_add
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          x✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ }) 0 x✝) 0
        -/
      add_zero := by
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          x✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ }) 0 x✝).f (CategoryTheory. …
        -/
        intros
        /-
          🎉 no goals
        -/
        apply Algebra.Hom.ext
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          ⊢ ∀ (n : Nat) (x : Quiver.Hom A₁ A₂), Eq ((fun n α => { f := HSMul.hSMul n α.f …
        -/
        apply add_zero
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          n✝ : Nat
          x✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ }) (HAdd.hAdd n✝ 1) x✝) (HA …
        -/
      nsmul_zero := by
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          n✝ : Nat
          x✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ }) (HAdd.hAdd n✝ 1) x✝).f ( …
        -/
        intros
        /-
          🎉 no goals
        -/
        apply Algebra.Hom.ext
        apply zero_smul
      nsmul_succ := by
        intros
        apply Algebra.Hom.ext
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          ⊢ ∀ (a b : Quiver.Hom A₁ A₂), Eq (HSub.hSub a b) (HAdd.hAdd a (Neg.neg b))
        -/
        apply succ_nsmul
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          a✝ b✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HSub.hSub a✝ b✝) (HAdd.hAdd a✝ (Neg.neg b✝))
        -/
      sub_eq_add_neg := by
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          a✝ b✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HSub.hSub a✝ b✝).f (HAdd.hAdd a✝ (Neg.neg b✝)).f
        -/
        intros
        /-
          🎉 no goals
        -/
        apply Algebra.Hom.ext
        apply sub_eq_add_neg
      zsmul_zero' := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          ⊢ ∀ (a : Quiver.Hom A₁ A₂), Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ } …
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) 0 a✝) 0
        -/
        apply Algebra.Hom.ext
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) 0 a✝).f (CategoryTheory. …
        -/
        apply zero_smul
        /-
          🎉 no goals
        -/
      zsmul_succ' := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          ⊢ ∀ (n : Nat) (a : Quiver.Hom A₁ A₂), Eq ((fun r α => { f := HSMul.hSMul r α.f …
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          n✝ : Nat
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) (↑n✝.succ) a✝) (HAdd.hAd …
        -/
        apply Algebra.Hom.ext
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          n✝ : Nat
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) (↑n✝.succ) a✝).f (HAdd.h …
        -/
        simp only [natCast_zsmul, succ_nsmul]
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          n✝ : Nat
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd (HSMul.hSMul n✝ a✝.f) a✝.f) (HAdd.hAdd { f := HSMul.hSMul n✝ a …
        -/
        rfl
        /-
          🎉 no goals
        -/
      zsmul_neg' := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          ⊢ ∀ (n : Nat) (a : Quiver.Hom A₁ A₂), Eq ((fun r α => { f := HSMul.hSMul r α.f …
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          n✝ : Nat
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) (Int.negSucc n✝) a✝) (Ne …
        -/
        apply Algebra.Hom.ext
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          n✝ : Nat
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) (Int.negSucc n✝) a✝).f ( …
        -/
        simp only [negSucc_zsmul, neg_inj, ← Nat.cast_smul_eq_nsmul ℤ]
        /-
          🎉 no goals
        -/
      neg_add_cancel := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          ⊢ ∀ (a : Quiver.Hom A₁ A₂), Eq (HAdd.hAdd (Neg.neg a) a) 0
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd (Neg.neg a✝) a✝) 0
        -/
        apply Algebra.Hom.ext
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd (Neg.neg a✝) a✝).f (CategoryTheory.Endofunctor.Algebra.Hom.f 0)
        -/
        apply neg_add_cancel
        /-
          🎉 no goals
        -/
      add_comm := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          ⊢ ∀ (a b : Quiver.Hom A₁ A₂), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          a✝ b✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd a✝ b✝) (HAdd.hAdd b✝ a✝)
        -/
        apply Algebra.Hom.ext
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          a✝ b✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd a✝ b✝).f (HAdd.hAdd b✝ a✝).f
        -/
        apply add_comm }
        /-
          🎉 no goals
        -/
  add_comp := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      F : CategoryTheory.Functor C C
      inst✝ : F.Additive
      ⊢ ∀ (P Q R : CategoryTheory.Endofunctor.Algebra F) (f f' : Quiver.Hom P Q) (g  …
    -/
    intros
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      F : CategoryTheory.Functor C C
      inst✝ : F.Additive
      P✝ Q✝ R✝ : CategoryTheory.Endofunctor.Algebra F
      f✝ f'✝ : Quiver.Hom P✝ Q✝
      g✝ : Quiver.Hom Q✝ R✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd f✝ f'✝) g✝) (HAdd.hAdd (Ca …
    -/
    apply Algebra.Hom.ext
    /-
      case f
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      F : CategoryTheory.Functor C C
      inst✝ : F.Additive
      P✝ Q✝ R✝ : CategoryTheory.Endofunctor.Algebra F
      f✝ f'✝ : Quiver.Hom P✝ Q✝
      g✝ : Quiver.Hom Q✝ R✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd f✝ f'✝) g✝).f (HAdd.hAdd ( …
    -/
    apply add_comp
    /-
      🎉 no goals
    -/
  comp_add := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      F : CategoryTheory.Functor C C
      inst✝ : F.Additive
      ⊢ ∀ (P Q R : CategoryTheory.Endofunctor.Algebra F) (f : Quiver.Hom P Q) (g g'  …
    -/
    intros
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      F : CategoryTheory.Functor C C
      inst✝ : F.Additive
      P✝ Q✝ R✝ : CategoryTheory.Endofunctor.Algebra F
      f✝ : Quiver.Hom P✝ Q✝
      g✝ g'✝ : Quiver.Hom Q✝ R✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f✝ (HAdd.hAdd g✝ g'✝)) (HAdd.hAdd (Ca …
    -/
    apply Algebra.Hom.ext
    /-
      case f
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      F : CategoryTheory.Functor C C
      inst✝ : F.Additive
      P✝ Q✝ R✝ : CategoryTheory.Endofunctor.Algebra F
      f✝ : Quiver.Hom P✝ Q✝
      g✝ g'✝ : Quiver.Hom Q✝ R✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f✝ (HAdd.hAdd g✝ g'✝)).f (HAdd.hAdd ( …
    -/
    apply comp_add
    /-
      🎉 no goals
    -/


instance Algebra.forget_additive : (Endofunctor.Algebra.forget F).Additive where


@[simps]
instance Endofunctor.coalgebraPreadditive : Preadditive (Endofunctor.Coalgebra F) where
  homGroup A₁ A₂ :=
    { add := fun α β =>
        { f := α.f + β.f
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    inst✝¹ : CategoryTheory.Preadditive C
                    F : CategoryTheory.Functor C C
                    inst✝ : F.Additive
                    A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
                    α β : Quiver.Hom A₁ A₂
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp A₁.str (F.map (HAdd.hAdd α.f β.f))) ( …
                  -/
          h := by simp only [Functor.map_add, comp_add, Endofunctor.Coalgebra.Hom.h, add_comp] }
                  /-
                    🎉 no goals
                  -/
      zero :=
        { f := 0
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    inst✝¹ : CategoryTheory.Preadditive C
                    F : CategoryTheory.Functor C C
                    inst✝ : F.Additive
                    A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp A₁.str (F.map 0)) (CategoryTheory.Cat …
                  -/
          h := by simp only [Functor.map_zero, zero_comp, comp_zero] }
                  /-
                    🎉 no goals
                  -/
      nsmul := fun n α =>
        { f := n • α.f
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    inst✝¹ : CategoryTheory.Preadditive C
                    F : CategoryTheory.Functor C C
                    inst✝ : F.Additive
                    A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
                    n : Nat
                    α : Quiver.Hom A₁ A₂
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp A₁.str (F.map (HSMul.hSMul n α.f))) ( …
                  -/
          h := by rw [Functor.map_nsmul, comp_nsmul, Endofunctor.Coalgebra.Hom.h, nsmul_comp] }
                  /-
                    🎉 no goals
                  -/
      neg := fun α =>
        { f := -α.f
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    inst✝¹ : CategoryTheory.Preadditive C
                    F : CategoryTheory.Functor C C
                    inst✝ : F.Additive
                    A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
                    α : Quiver.Hom A₁ A₂
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp A₁.str (F.map (Neg.neg α.f))) (Catego …
                  -/
          h := by simp only [Functor.map_neg, comp_neg, Endofunctor.Coalgebra.Hom.h, neg_comp] }
                  /-
                    🎉 no goals
                  -/
      sub := fun α β =>
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          ⊢ ∀ (a b c : Quiver.Hom A₁ A₂), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a  …
        -/
        { f := α.f - β.f
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          a✝ b✝ c✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd (HAdd.hAdd a✝ b✝) c✝) (HAdd.hAdd a✝ (HAdd.hAdd b✝ c✝))
        -/
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    inst✝¹ : CategoryTheory.Preadditive C
                    F : CategoryTheory.Functor C C
                    inst✝ : F.Additive
                    A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
                    α β : Quiver.Hom A₁ A₂
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp A₁.str (F.map (HSub.hSub α.f β.f))) ( …
                  -/
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          a✝ b✝ c✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd (HAdd.hAdd a✝ b✝) c✝).f (HAdd.hAdd a✝ (HAdd.hAdd b✝ c✝)).f
        -/
          h := by simp only [Functor.map_sub, comp_sub, Endofunctor.Coalgebra.Hom.h, sub_comp] }
        /-
          🎉 no goals
        -/
                  /-
                    🎉 no goals
                  -/
      zsmul := fun r α =>
        { f := r • α.f
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          ⊢ ∀ (a : Quiver.Hom A₁ A₂), Eq (HAdd.hAdd 0 a) a
        -/
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    inst✝¹ : CategoryTheory.Preadditive C
                    F : CategoryTheory.Functor C C
                    inst✝ : F.Additive
                    A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
                    r : Int
                    α : Quiver.Hom A₁ A₂
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp A₁.str (F.map (HSMul.hSMul r α.f))) ( …
                  -/
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd 0 a✝) a✝
        -/
          h := by rw [Functor.map_zsmul, comp_zsmul, Endofunctor.Coalgebra.Hom.h, zsmul_comp] }
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd 0 a✝).f a✝.f
        -/
                  /-
                    🎉 no goals
                  -/
        /-
          🎉 no goals
        -/
      add_assoc := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          ⊢ ∀ (a : Quiver.Hom A₁ A₂), Eq (HAdd.hAdd a 0) a
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd a✝ 0) a✝
        -/
        apply Coalgebra.Hom.ext
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd a✝ 0).f a✝.f
        -/
        apply add_assoc
        /-
          🎉 no goals
        -/
      zero_add := by
        intros
        apply Coalgebra.Hom.ext
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          ⊢ ∀ (x : Quiver.Hom A₁ A₂), Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ } …
        -/
        apply zero_add
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          x✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ }) 0 x✝) 0
        -/
      add_zero := by
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          x✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ }) 0 x✝).f (CategoryTheory. …
        -/
        intros
        /-
          🎉 no goals
        -/
        apply Coalgebra.Hom.ext
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          ⊢ ∀ (n : Nat) (x : Quiver.Hom A₁ A₂), Eq ((fun n α => { f := HSMul.hSMul n α.f …
        -/
        apply add_zero
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          n✝ : Nat
          x✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ }) (HAdd.hAdd n✝ 1) x✝) (HA …
        -/
      nsmul_zero := by
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          n✝ : Nat
          x✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ }) (HAdd.hAdd n✝ 1) x✝).f ( …
        -/
        intros
        /-
          🎉 no goals
        -/
        apply Coalgebra.Hom.ext
        apply zero_smul
      nsmul_succ := by
        intros
        apply Coalgebra.Hom.ext
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          ⊢ ∀ (a b : Quiver.Hom A₁ A₂), Eq (HSub.hSub a b) (HAdd.hAdd a (Neg.neg b))
        -/
        apply succ_nsmul
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          a✝ b✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HSub.hSub a✝ b✝) (HAdd.hAdd a✝ (Neg.neg b✝))
        -/
      sub_eq_add_neg := by
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          a✝ b✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HSub.hSub a✝ b✝).f (HAdd.hAdd a✝ (Neg.neg b✝)).f
        -/
        intros
        /-
          🎉 no goals
        -/
        apply Coalgebra.Hom.ext
        apply sub_eq_add_neg
      zsmul_zero' := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          ⊢ ∀ (a : Quiver.Hom A₁ A₂), Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ } …
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) 0 a✝) 0
        -/
        apply Coalgebra.Hom.ext
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) 0 a✝).f (CategoryTheory. …
        -/
        apply zero_smul
        /-
          🎉 no goals
        -/
      zsmul_succ' := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          ⊢ ∀ (n : Nat) (a : Quiver.Hom A₁ A₂), Eq ((fun r α => { f := HSMul.hSMul r α.f …
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          n✝ : Nat
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) (↑n✝.succ) a✝) (HAdd.hAd …
        -/
        apply Coalgebra.Hom.ext
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          n✝ : Nat
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) (↑n✝.succ) a✝).f (HAdd.h …
        -/
        simp only [natCast_zsmul, succ_nsmul]
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          n✝ : Nat
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd (HSMul.hSMul n✝ a✝.f) a✝.f) (HAdd.hAdd { f := HSMul.hSMul n✝ a …
        -/
        rfl
        /-
          🎉 no goals
        -/
      zsmul_neg' := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          ⊢ ∀ (n : Nat) (a : Quiver.Hom A₁ A₂), Eq ((fun r α => { f := HSMul.hSMul r α.f …
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          n✝ : Nat
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) (Int.negSucc n✝) a✝) (Ne …
        -/
        apply Coalgebra.Hom.ext
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          n✝ : Nat
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) (Int.negSucc n✝) a✝).f ( …
        -/
        simp only [negSucc_zsmul, neg_inj, ← Nat.cast_smul_eq_nsmul ℤ]
        /-
          🎉 no goals
        -/
      neg_add_cancel := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          ⊢ ∀ (a : Quiver.Hom A₁ A₂), Eq (HAdd.hAdd (Neg.neg a) a) 0
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd (Neg.neg a✝) a✝) 0
        -/
        apply Coalgebra.Hom.ext
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          a✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd (Neg.neg a✝) a✝).f (CategoryTheory.Endofunctor.Coalgebra.Hom.f …
        -/
        apply neg_add_cancel
        /-
          🎉 no goals
        -/
      add_comm := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          ⊢ ∀ (a b : Quiver.Hom A₁ A₂), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          a✝ b✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd a✝ b✝) (HAdd.hAdd b✝ a✝)
        -/
        apply Coalgebra.Hom.ext
        /-
          case f
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          F : CategoryTheory.Functor C C
          inst✝ : F.Additive
          A₁ A₂ : CategoryTheory.Endofunctor.Coalgebra F
          a✝ b✝ : Quiver.Hom A₁ A₂
          ⊢ Eq (HAdd.hAdd a✝ b✝).f (HAdd.hAdd b✝ a✝).f
        -/
        apply add_comm }
        /-
          🎉 no goals
        -/
  add_comp := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      F : CategoryTheory.Functor C C
      inst✝ : F.Additive
      ⊢ ∀ (P Q R : CategoryTheory.Endofunctor.Coalgebra F) (f f' : Quiver.Hom P Q) ( …
    -/
    intros
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      F : CategoryTheory.Functor C C
      inst✝ : F.Additive
      P✝ Q✝ R✝ : CategoryTheory.Endofunctor.Coalgebra F
      f✝ f'✝ : Quiver.Hom P✝ Q✝
      g✝ : Quiver.Hom Q✝ R✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd f✝ f'✝) g✝) (HAdd.hAdd (Ca …
    -/
    apply Coalgebra.Hom.ext
    /-
      case f
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      F : CategoryTheory.Functor C C
      inst✝ : F.Additive
      P✝ Q✝ R✝ : CategoryTheory.Endofunctor.Coalgebra F
      f✝ f'✝ : Quiver.Hom P✝ Q✝
      g✝ : Quiver.Hom Q✝ R✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd f✝ f'✝) g✝).f (HAdd.hAdd ( …
    -/
    apply add_comp
    /-
      🎉 no goals
    -/
  comp_add := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      F : CategoryTheory.Functor C C
      inst✝ : F.Additive
      ⊢ ∀ (P Q R : CategoryTheory.Endofunctor.Coalgebra F) (f : Quiver.Hom P Q) (g g …
    -/
    intros
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      F : CategoryTheory.Functor C C
      inst✝ : F.Additive
      P✝ Q✝ R✝ : CategoryTheory.Endofunctor.Coalgebra F
      f✝ : Quiver.Hom P✝ Q✝
      g✝ g'✝ : Quiver.Hom Q✝ R✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f✝ (HAdd.hAdd g✝ g'✝)) (HAdd.hAdd (Ca …
    -/
    apply Coalgebra.Hom.ext
    /-
      case f
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      F : CategoryTheory.Functor C C
      inst✝ : F.Additive
      P✝ Q✝ R✝ : CategoryTheory.Endofunctor.Coalgebra F
      f✝ : Quiver.Hom P✝ Q✝
      g✝ g'✝ : Quiver.Hom Q✝ R✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f✝ (HAdd.hAdd g✝ g'✝)).f (HAdd.hAdd ( …
    -/
    apply comp_add
    /-
      🎉 no goals
    -/


instance Coalgebra.forget_additive : (Endofunctor.Coalgebra.forget F).Additive where


