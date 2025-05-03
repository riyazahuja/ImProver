/-- The category of algebras over an additive monad on a preadditive category is preadditive. -/
@[simps]
instance Monad.algebraPreadditive : Preadditive (Monad.Algebra T) where
  homGroup F G :=
    { add := fun α β =>
        { f := α.f + β.f
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    inst✝¹ : CategoryTheory.Preadditive C
                    T : CategoryTheory.Monad C
                    inst✝ : T.Additive
                    F G : T.Algebra
                    α β : Quiver.Hom F G
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map (HAdd.hAdd α.f β.f)) G.a) (Cat …
                  -/
          h := by simp only [Functor.map_add, add_comp, Monad.Algebra.Hom.h, comp_add] }
                  /-
                    🎉 no goals
                  -/
      zero :=
        { f := 0
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    inst✝¹ : CategoryTheory.Preadditive C
                    T : CategoryTheory.Monad C
                    inst✝ : T.Additive
                    F G : T.Algebra
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map 0) G.a) (CategoryTheory.Catego …
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
                    T : CategoryTheory.Monad C
                    inst✝ : T.Additive
                    F G : T.Algebra
                    n : Nat
                    α : Quiver.Hom F G
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map (HSMul.hSMul n α.f)) G.a) (Cat …
                  -/
          h := by rw [Functor.map_nsmul, nsmul_comp, Monad.Algebra.Hom.h, comp_nsmul] }
                  /-
                    🎉 no goals
                  -/
      neg := fun α =>
        { f := -α.f
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    inst✝¹ : CategoryTheory.Preadditive C
                    T : CategoryTheory.Monad C
                    inst✝ : T.Additive
                    F G : T.Algebra
                    α : Quiver.Hom F G
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map (Neg.neg α.f)) G.a) (CategoryT …
                  -/
          h := by simp only [Functor.map_neg, neg_comp, Monad.Algebra.Hom.h, comp_neg] }
                  /-
                    🎉 no goals
                  -/
      sub := fun α β =>
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          ⊢ ∀ (a b c : Quiver.Hom F G), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (H …
        -/
        { f := α.f - β.f
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          a✝ b✝ c✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd (HAdd.hAdd a✝ b✝) c✝) (HAdd.hAdd a✝ (HAdd.hAdd b✝ c✝))
        -/
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    inst✝¹ : CategoryTheory.Preadditive C
                    T : CategoryTheory.Monad C
                    inst✝ : T.Additive
                    F G : T.Algebra
                    α β : Quiver.Hom F G
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map (HSub.hSub α.f β.f)) G.a) (Cat …
                  -/
        /-
          case h
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          a✝ b✝ c✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd (HAdd.hAdd a✝ b✝) c✝).f (HAdd.hAdd a✝ (HAdd.hAdd b✝ c✝)).f
        -/
          h := by simp only [Functor.map_sub, sub_comp, Monad.Algebra.Hom.h, comp_sub] }
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
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          ⊢ ∀ (a : Quiver.Hom F G), Eq (HAdd.hAdd 0 a) a
        -/
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    inst✝¹ : CategoryTheory.Preadditive C
                    T : CategoryTheory.Monad C
                    inst✝ : T.Additive
                    F G : T.Algebra
                    r : Int
                    α : Quiver.Hom F G
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map (HSMul.hSMul r α.f)) G.a) (Cat …
                  -/
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          a✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd 0 a✝) a✝
        -/
          h := by rw [Functor.map_zsmul, zsmul_comp, Monad.Algebra.Hom.h, comp_zsmul] }
        /-
          case h
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          a✝ : Quiver.Hom F G
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
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          ⊢ ∀ (a : Quiver.Hom F G), Eq (HAdd.hAdd a 0) a
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          a✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd a✝ 0) a✝
        -/
        ext
        /-
          case h
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          a✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd a✝ 0).f a✝.f
        -/
        apply add_assoc
        /-
          🎉 no goals
        -/
      zero_add := by
        intros
        ext
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          ⊢ ∀ (x : Quiver.Hom F G), Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ })  …
        -/
        apply zero_add
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          x✝ : Quiver.Hom F G
          ⊢ Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ }) 0 x✝) 0
        -/
      add_zero := by
        /-
          case h
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          x✝ : Quiver.Hom F G
          ⊢ Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ }) 0 x✝).f (CategoryTheory. …
        -/
        intros
        /-
          🎉 no goals
        -/
        ext
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          ⊢ ∀ (n : Nat) (x : Quiver.Hom F G), Eq ((fun n α => { f := HSMul.hSMul n α.f,  …
        -/
        apply add_zero
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          n✝ : Nat
          x✝ : Quiver.Hom F G
          ⊢ Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ }) (HAdd.hAdd n✝ 1) x✝) (HA …
        -/
      nsmul_zero := by
        /-
          case h
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          n✝ : Nat
          x✝ : Quiver.Hom F G
          ⊢ Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ }) (HAdd.hAdd n✝ 1) x✝).f ( …
        -/
        intros
        /-
          🎉 no goals
        -/
        ext
        apply zero_smul
      nsmul_succ := by
        intros
        ext
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          ⊢ ∀ (a b : Quiver.Hom F G), Eq (HSub.hSub a b) (HAdd.hAdd a (Neg.neg b))
        -/
        apply succ_nsmul
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          a✝ b✝ : Quiver.Hom F G
          ⊢ Eq (HSub.hSub a✝ b✝) (HAdd.hAdd a✝ (Neg.neg b✝))
        -/
      sub_eq_add_neg := by
        /-
          case h
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          a✝ b✝ : Quiver.Hom F G
          ⊢ Eq (HSub.hSub a✝ b✝).f (HAdd.hAdd a✝ (Neg.neg b✝)).f
        -/
        intros
        /-
          🎉 no goals
        -/
        ext
        apply sub_eq_add_neg
      zsmul_zero' := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          ⊢ ∀ (a : Quiver.Hom F G), Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ })  …
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          a✝ : Quiver.Hom F G
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) 0 a✝) 0
        -/
        ext
        /-
          case h
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          a✝ : Quiver.Hom F G
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
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          ⊢ ∀ (n : Nat) (a : Quiver.Hom F G), Eq ((fun r α => { f := HSMul.hSMul r α.f,  …
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          n✝ : Nat
          a✝ : Quiver.Hom F G
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) (↑n✝.succ) a✝) (HAdd.hAd …
        -/
        ext
        /-
          case h
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          n✝ : Nat
          a✝ : Quiver.Hom F G
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) (↑n✝.succ) a✝).f (HAdd.h …
        -/
        simp only [natCast_zsmul, succ_nsmul]
        /-
          case h
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          n✝ : Nat
          a✝ : Quiver.Hom F G
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
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          ⊢ ∀ (n : Nat) (a : Quiver.Hom F G), Eq ((fun r α => { f := HSMul.hSMul r α.f,  …
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          n✝ : Nat
          a✝ : Quiver.Hom F G
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) (Int.negSucc n✝) a✝) (Ne …
        -/
        ext
        /-
          case h
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          n✝ : Nat
          a✝ : Quiver.Hom F G
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
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          ⊢ ∀ (a : Quiver.Hom F G), Eq (HAdd.hAdd (Neg.neg a) a) 0
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          a✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd (Neg.neg a✝) a✝) 0
        -/
        ext
        /-
          case h
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          a✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd (Neg.neg a✝) a✝).f (CategoryTheory.Monad.Algebra.Hom.f 0)
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
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          ⊢ ∀ (a b : Quiver.Hom F G), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          a✝ b✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd a✝ b✝) (HAdd.hAdd b✝ a✝)
        -/
        ext
        /-
          case h
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝ : T.Additive
          F G : T.Algebra
          a✝ b✝ : Quiver.Hom F G
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
      T : CategoryTheory.Monad C
      inst✝ : T.Additive
      ⊢ ∀ (P Q R : T.Algebra) (f f' : Quiver.Hom P Q) (g : Quiver.Hom Q R), Eq (Cate …
    -/
    intros
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      T : CategoryTheory.Monad C
      inst✝ : T.Additive
      P✝ Q✝ R✝ : T.Algebra
      f✝ f'✝ : Quiver.Hom P✝ Q✝
      g✝ : Quiver.Hom Q✝ R✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd f✝ f'✝) g✝) (HAdd.hAdd (Ca …
    -/
    ext
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      T : CategoryTheory.Monad C
      inst✝ : T.Additive
      P✝ Q✝ R✝ : T.Algebra
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
      T : CategoryTheory.Monad C
      inst✝ : T.Additive
      ⊢ ∀ (P Q R : T.Algebra) (f : Quiver.Hom P Q) (g g' : Quiver.Hom Q R), Eq (Cate …
    -/
    intros
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      T : CategoryTheory.Monad C
      inst✝ : T.Additive
      P✝ Q✝ R✝ : T.Algebra
      f✝ : Quiver.Hom P✝ Q✝
      g✝ g'✝ : Quiver.Hom Q✝ R✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f✝ (HAdd.hAdd g✝ g'✝)) (HAdd.hAdd (Ca …
    -/
    ext
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      T : CategoryTheory.Monad C
      inst✝ : T.Additive
      P✝ Q✝ R✝ : T.Algebra
      f✝ : Quiver.Hom P✝ Q✝
      g✝ g'✝ : Quiver.Hom Q✝ R✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f✝ (HAdd.hAdd g✝ g'✝)).f (HAdd.hAdd ( …
    -/
    apply comp_add
    /-
      🎉 no goals
    -/


instance Monad.forget_additive : (Monad.forget T).Additive where


/-- The category of coalgebras over an additive comonad on a preadditive category is preadditive. -/
@[simps]
instance Comonad.coalgebraPreadditive : Preadditive (Comonad.Coalgebra U) where
  homGroup F G :=
    { add := fun α β =>
        { f := α.f + β.f
                  /-
                    C : Type u₁
                    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                    inst✝² : CategoryTheory.Preadditive C
                    T : CategoryTheory.Monad C
                    inst✝¹ : T.Additive
                    U : CategoryTheory.Comonad C
                    inst✝ : U.Additive
                    F G : U.Coalgebra
                    α β : Quiver.Hom F G
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp F.a (U.map (HAdd.hAdd α.f β.f))) (Cat …
                  -/
          h := by simp only [Functor.map_add, comp_add, Comonad.Coalgebra.Hom.h, add_comp] }
                  /-
                    🎉 no goals
                  -/
      zero :=
        { f := 0
                  /-
                    C : Type u₁
                    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                    inst✝² : CategoryTheory.Preadditive C
                    T : CategoryTheory.Monad C
                    inst✝¹ : T.Additive
                    U : CategoryTheory.Comonad C
                    inst✝ : U.Additive
                    F G : U.Coalgebra
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp F.a (U.map 0)) (CategoryTheory.Catego …
                  -/
          h := by simp only [Functor.map_zero, comp_zero, zero_comp] }
                  /-
                    🎉 no goals
                  -/
      nsmul := fun n α =>
        { f := n • α.f
                  /-
                    C : Type u₁
                    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                    inst✝² : CategoryTheory.Preadditive C
                    T : CategoryTheory.Monad C
                    inst✝¹ : T.Additive
                    U : CategoryTheory.Comonad C
                    inst✝ : U.Additive
                    F G : U.Coalgebra
                    n : Nat
                    α : Quiver.Hom F G
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp F.a (U.map (HSMul.hSMul n α.f))) (Cat …
                  -/
          h := by rw [Functor.map_nsmul, comp_nsmul, Comonad.Coalgebra.Hom.h, nsmul_comp] }
                  /-
                    🎉 no goals
                  -/
      neg := fun α =>
        { f := -α.f
                  /-
                    C : Type u₁
                    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                    inst✝² : CategoryTheory.Preadditive C
                    T : CategoryTheory.Monad C
                    inst✝¹ : T.Additive
                    U : CategoryTheory.Comonad C
                    inst✝ : U.Additive
                    F G : U.Coalgebra
                    α : Quiver.Hom F G
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp F.a (U.map (Neg.neg α.f))) (CategoryT …
                  -/
          h := by simp only [Functor.map_neg, comp_neg, Comonad.Coalgebra.Hom.h, neg_comp] }
                  /-
                    🎉 no goals
                  -/
      sub := fun α β =>
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          ⊢ ∀ (a b c : Quiver.Hom F G), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (H …
        -/
        { f := α.f - β.f
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          a✝ b✝ c✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd (HAdd.hAdd a✝ b✝) c✝) (HAdd.hAdd a✝ (HAdd.hAdd b✝ c✝))
        -/
                  /-
                    C : Type u₁
                    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                    inst✝² : CategoryTheory.Preadditive C
                    T : CategoryTheory.Monad C
                    inst✝¹ : T.Additive
                    U : CategoryTheory.Comonad C
                    inst✝ : U.Additive
                    F G : U.Coalgebra
                    α β : Quiver.Hom F G
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp F.a (U.map (HSub.hSub α.f β.f))) (Cat …
                  -/
        /-
          case h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          a✝ b✝ c✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd (HAdd.hAdd a✝ b✝) c✝).f (HAdd.hAdd a✝ (HAdd.hAdd b✝ c✝)).f
        -/
          h := by simp only [Functor.map_sub, comp_sub, Comonad.Coalgebra.Hom.h, sub_comp] }
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
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          ⊢ ∀ (a : Quiver.Hom F G), Eq (HAdd.hAdd 0 a) a
        -/
                  /-
                    C : Type u₁
                    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                    inst✝² : CategoryTheory.Preadditive C
                    T : CategoryTheory.Monad C
                    inst✝¹ : T.Additive
                    U : CategoryTheory.Comonad C
                    inst✝ : U.Additive
                    F G : U.Coalgebra
                    r : Int
                    α : Quiver.Hom F G
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp F.a (U.map (HSMul.hSMul r α.f))) (Cat …
                  -/
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          a✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd 0 a✝) a✝
        -/
          h := by rw [Functor.map_zsmul, comp_zsmul, Comonad.Coalgebra.Hom.h, zsmul_comp] }
        /-
          case h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          a✝ : Quiver.Hom F G
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
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          ⊢ ∀ (a : Quiver.Hom F G), Eq (HAdd.hAdd a 0) a
        -/
        intros
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          a✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd a✝ 0) a✝
        -/
        ext
        /-
          case h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          a✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd a✝ 0).f a✝.f
        -/
        apply add_assoc
        /-
          🎉 no goals
        -/
      zero_add := by
        intros
        ext
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          ⊢ ∀ (x : Quiver.Hom F G), Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ })  …
        -/
        apply zero_add
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          x✝ : Quiver.Hom F G
          ⊢ Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ }) 0 x✝) 0
        -/
      add_zero := by
        /-
          case h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          x✝ : Quiver.Hom F G
          ⊢ Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ }) 0 x✝).f (CategoryTheory. …
        -/
        intros
        /-
          🎉 no goals
        -/
        ext
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          ⊢ ∀ (n : Nat) (x : Quiver.Hom F G), Eq ((fun n α => { f := HSMul.hSMul n α.f,  …
        -/
        apply add_zero
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          n✝ : Nat
          x✝ : Quiver.Hom F G
          ⊢ Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ }) (HAdd.hAdd n✝ 1) x✝) (HA …
        -/
      nsmul_zero := by
        /-
          case h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          n✝ : Nat
          x✝ : Quiver.Hom F G
          ⊢ Eq ((fun n α => { f := HSMul.hSMul n α.f, h := ⋯ }) (HAdd.hAdd n✝ 1) x✝).f ( …
        -/
        intros
        /-
          🎉 no goals
        -/
        ext
        apply zero_smul
      nsmul_succ := by
        intros
        ext
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          ⊢ ∀ (a b : Quiver.Hom F G), Eq (HSub.hSub a b) (HAdd.hAdd a (Neg.neg b))
        -/
        apply succ_nsmul
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          a✝ b✝ : Quiver.Hom F G
          ⊢ Eq (HSub.hSub a✝ b✝) (HAdd.hAdd a✝ (Neg.neg b✝))
        -/
      sub_eq_add_neg := by
        /-
          case h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          a✝ b✝ : Quiver.Hom F G
          ⊢ Eq (HSub.hSub a✝ b✝).f (HAdd.hAdd a✝ (Neg.neg b✝)).f
        -/
        intros
        /-
          🎉 no goals
        -/
        ext
        apply sub_eq_add_neg
      zsmul_zero' := by
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          ⊢ ∀ (a : Quiver.Hom F G), Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ })  …
        -/
        intros
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          a✝ : Quiver.Hom F G
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) 0 a✝) 0
        -/
        ext
        /-
          case h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          a✝ : Quiver.Hom F G
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) 0 a✝).f (CategoryTheory. …
        -/
        apply zero_smul
        /-
          🎉 no goals
        -/
      zsmul_succ' := by
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          ⊢ ∀ (n : Nat) (a : Quiver.Hom F G), Eq ((fun r α => { f := HSMul.hSMul r α.f,  …
        -/
        intros
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          n✝ : Nat
          a✝ : Quiver.Hom F G
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) (↑n✝.succ) a✝) (HAdd.hAd …
        -/
        ext
        /-
          case h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          n✝ : Nat
          a✝ : Quiver.Hom F G
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) (↑n✝.succ) a✝).f (HAdd.h …
        -/
        simp only [natCast_zsmul, succ_nsmul]
        /-
          case h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          n✝ : Nat
          a✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd (HSMul.hSMul n✝ a✝.f) a✝.f) (HAdd.hAdd { f := HSMul.hSMul n✝ a …
        -/
        rfl
        /-
          🎉 no goals
        -/
      zsmul_neg' := by
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          ⊢ ∀ (n : Nat) (a : Quiver.Hom F G), Eq ((fun r α => { f := HSMul.hSMul r α.f,  …
        -/
        intros
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          n✝ : Nat
          a✝ : Quiver.Hom F G
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) (Int.negSucc n✝) a✝) (Ne …
        -/
        ext
        /-
          case h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          n✝ : Nat
          a✝ : Quiver.Hom F G
          ⊢ Eq ((fun r α => { f := HSMul.hSMul r α.f, h := ⋯ }) (Int.negSucc n✝) a✝).f ( …
        -/
        simp only [negSucc_zsmul, neg_inj, ← Nat.cast_smul_eq_nsmul ℤ]
        /-
          🎉 no goals
        -/
      neg_add_cancel := by
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          ⊢ ∀ (a : Quiver.Hom F G), Eq (HAdd.hAdd (Neg.neg a) a) 0
        -/
        intros
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          a✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd (Neg.neg a✝) a✝) 0
        -/
        ext
        /-
          case h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          a✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd (Neg.neg a✝) a✝).f (CategoryTheory.Comonad.Coalgebra.Hom.f 0)
        -/
        apply neg_add_cancel
        /-
          🎉 no goals
        -/
      add_comm := by
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          ⊢ ∀ (a b : Quiver.Hom F G), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
        -/
        intros
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          a✝ b✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd a✝ b✝) (HAdd.hAdd b✝ a✝)
        -/
        ext
        /-
          case h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Preadditive C
          T : CategoryTheory.Monad C
          inst✝¹ : T.Additive
          U : CategoryTheory.Comonad C
          inst✝ : U.Additive
          F G : U.Coalgebra
          a✝ b✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd a✝ b✝).f (HAdd.hAdd b✝ a✝).f
        -/
        apply add_comm }
        /-
          🎉 no goals
        -/
  add_comp := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Preadditive C
      T : CategoryTheory.Monad C
      inst✝¹ : T.Additive
      U : CategoryTheory.Comonad C
      inst✝ : U.Additive
      ⊢ ∀ (P Q R : U.Coalgebra) (f f' : Quiver.Hom P Q) (g : Quiver.Hom Q R), Eq (Ca …
    -/
    intros
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Preadditive C
      T : CategoryTheory.Monad C
      inst✝¹ : T.Additive
      U : CategoryTheory.Comonad C
      inst✝ : U.Additive
      P✝ Q✝ R✝ : U.Coalgebra
      f✝ f'✝ : Quiver.Hom P✝ Q✝
      g✝ : Quiver.Hom Q✝ R✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd f✝ f'✝) g✝) (HAdd.hAdd (Ca …
    -/
    ext
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Preadditive C
      T : CategoryTheory.Monad C
      inst✝¹ : T.Additive
      U : CategoryTheory.Comonad C
      inst✝ : U.Additive
      P✝ Q✝ R✝ : U.Coalgebra
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
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Preadditive C
      T : CategoryTheory.Monad C
      inst✝¹ : T.Additive
      U : CategoryTheory.Comonad C
      inst✝ : U.Additive
      ⊢ ∀ (P Q R : U.Coalgebra) (f : Quiver.Hom P Q) (g g' : Quiver.Hom Q R), Eq (Ca …
    -/
    intros
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Preadditive C
      T : CategoryTheory.Monad C
      inst✝¹ : T.Additive
      U : CategoryTheory.Comonad C
      inst✝ : U.Additive
      P✝ Q✝ R✝ : U.Coalgebra
      f✝ : Quiver.Hom P✝ Q✝
      g✝ g'✝ : Quiver.Hom Q✝ R✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f✝ (HAdd.hAdd g✝ g'✝)) (HAdd.hAdd (Ca …
    -/
    ext
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Preadditive C
      T : CategoryTheory.Monad C
      inst✝¹ : T.Additive
      U : CategoryTheory.Comonad C
      inst✝ : U.Additive
      P✝ Q✝ R✝ : U.Coalgebra
      f✝ : Quiver.Hom P✝ Q✝
      g✝ g'✝ : Quiver.Hom Q✝ R✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f✝ (HAdd.hAdd g✝ g'✝)).f (HAdd.hAdd ( …
    -/
    apply comp_add
    /-
      🎉 no goals
    -/


instance Comonad.forget_additive : (Comonad.forget U).Additive where


