/-- The map `Cochain K L n → Cochain K (L⟦a⟧) n'` when `n' + a = n`. -/
def rightShift (a n' : ℤ) (hn' : n' + a = n) : Cochain K (L⟦a⟧) n' :=
  Cochain.mk (fun p q hpq => γ.v p (p + n) rfl ≫
                                           /-
                                             C : Type u
                                             inst✝³ : CategoryTheory.Category.{v, u} C
                                             inst✝² : CategoryTheory.Preadditive C
                                             R : Type u_1
                                             inst✝¹ : Ring R
                                             inst✝ : CategoryTheory.Linear R C
                                             K L M : CochainComplex C Int
                                             n : Int
                                             γ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n
                                             a n' : Int
                                             hn' : Eq (HAdd.hAdd n' a) n
                                             p q : Int
                                             hpq : Eq (HAdd.hAdd p n') q
                                             ⊢ Eq (HAdd.hAdd p n) (HAdd.hAdd q a)
                                           -/
    (L.shiftFunctorObjXIso a q (p + n) (by omega)).inv)
                                           /-
                                             🎉 no goals
                                           -/


lemma rightShift_v (a n' : ℤ) (hn' : n' + a = n) (p q : ℤ) (hpq : p + n' = q)
    (p' : ℤ) (hp' : p + n = p') :
    (γ.rightShift a n' hn').v p q hpq = γ.v p p' hp' ≫
                                        /-
                                          C : Type u
                                          inst✝³ : CategoryTheory.Category.{v, u} C
                                          inst✝² : CategoryTheory.Preadditive C
                                          R : Type u_1
                                          inst✝¹ : Ring R
                                          inst✝ : CategoryTheory.Linear R C
                                          K L M : CochainComplex C Int
                                          n : Int
                                          γ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n
                                          a n' : Int
                                          hn' : Eq (HAdd.hAdd n' a) n
                                          p q : Int
                                          hpq : Eq (HAdd.hAdd p n') q
                                          p' : Int
                                          hp' : Eq (HAdd.hAdd p n) p'
                                          ⊢ Eq p' (HAdd.hAdd q a)
                                        -/
      (L.shiftFunctorObjXIso a q p' (by rw [← hp', ← hpq, ← hn', add_assoc])).inv := by
                                        /-
                                          🎉 no goals
                                        -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    p q : Int
    hpq : Eq (HAdd.hAdd p n') q
    p' : Int
    hp' : Eq (HAdd.hAdd p n) p'
    ⊢ Eq ((γ.rightShift a n' hn').v p q hpq) (CategoryTheory.CategoryStruct.comp ( …
  -/
  subst hp'
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    p q : Int
    hpq : Eq (HAdd.hAdd p n') q
    ⊢ Eq ((γ.rightShift a n' hn').v p q hpq) (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp only [rightShift]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    p q : Int
    hpq : Eq (HAdd.hAdd p n') q
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.mk fun p q hpq => CategoryTheory.Cate …
  -/
  simp only [mk_v]
  /-
    🎉 no goals
  -/


/-- The map `Cochain K L n → Cochain (K⟦a⟧) L n'` when `n + a = n'`. -/
def leftShift (a n' : ℤ) (hn' : n + a = n') : Cochain (K⟦a⟧) L n' :=
  Cochain.mk (fun p q hpq => (a * n' + ((a * (a-1))/2)).negOnePow •
                                                                  /-
                                                                    C : Type u
                                                                    inst✝³ : CategoryTheory.Category.{v, u} C
                                                                    inst✝² : CategoryTheory.Preadditive C
                                                                    R : Type u_1
                                                                    inst✝¹ : Ring R
                                                                    inst✝ : CategoryTheory.Linear R C
                                                                    K L M : CochainComplex C Int
                                                                    n : Int
                                                                    γ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n
                                                                    a n' : Int
                                                                    hn' : Eq (HAdd.hAdd n a) n'
                                                                    p q : Int
                                                                    hpq : Eq (HAdd.hAdd p n') q
                                                                    ⊢ Eq (HAdd.hAdd (HAdd.hAdd p a) n) q
                                                                  -/
    (K.shiftFunctorObjXIso a p (p + a) rfl).hom ≫ γ.v (p+a) q (by omega))
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


lemma leftShift_v (a n' : ℤ) (hn' : n + a = n') (p q : ℤ) (hpq : p + n' = q)
    (p' : ℤ) (hp' : p' + n = q) :
    (γ.leftShift a n' hn').v p q hpq = (a * n' + ((a * (a - 1))/2)).negOnePow •
      (K.shiftFunctorObjXIso a p p'
            /-
              C : Type u
              inst✝³ : CategoryTheory.Category.{v, u} C
              inst✝² : CategoryTheory.Preadditive C
              R : Type u_1
              inst✝¹ : Ring R
              inst✝ : CategoryTheory.Linear R C
              K L M : CochainComplex C Int
              n : Int
              γ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n
              a n' : Int
              hn' : Eq (HAdd.hAdd n a) n'
              p q : Int
              hpq : Eq (HAdd.hAdd p n') q
              p' : Int
              hp' : Eq (HAdd.hAdd p' n) q
              ⊢ Eq p' (HAdd.hAdd p a)
            -/
        (by rw [← add_left_inj n, hp', add_assoc, add_comm a, hn', hpq])).hom ≫ γ.v p' q hp' := by
            /-
              🎉 no goals
            -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    p q : Int
    hpq : Eq (HAdd.hAdd p n') q
    p' : Int
    hp' : Eq (HAdd.hAdd p' n) q
    ⊢ Eq ((γ.leftShift a n' hn').v p q hpq) (HSMul.hSMul (HAdd.hAdd (HMul.hMul a n …
  -/
  obtain rfl : p' = p + a := by omega
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    p q : Int
    hpq : Eq (HAdd.hAdd p n') q
    hp' : Eq (HAdd.hAdd (HAdd.hAdd p a) n) q
    ⊢ Eq ((γ.leftShift a n' hn').v p q hpq) (HSMul.hSMul (HAdd.hAdd (HMul.hMul a n …
  -/
  dsimp only [leftShift]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    p q : Int
    hpq : Eq (HAdd.hAdd p n') q
    hp' : Eq (HAdd.hAdd (HAdd.hAdd p a) n) q
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.mk fun p q hpq => HSMul.hSMul (HAdd.h …
  -/
  simp only [mk_v]
  /-
    🎉 no goals
  -/


/-- The map `Cochain K (L⟦a⟧) n' → Cochain K L n` when `n' + a = n`. -/
def rightUnshift {n' a : ℤ} (γ : Cochain K (L⟦a⟧) n') (n : ℤ) (hn : n' + a = n) :
    Cochain K L n :=
  Cochain.mk (fun p q hpq => γ.v p (p + n') rfl ≫
                                            /-
                                              C : Type u
                                              inst✝³ : CategoryTheory.Category.{v, u} C
                                              inst✝² : CategoryTheory.Preadditive C
                                              R : Type u_1
                                              inst✝¹ : Ring R
                                              inst✝ : CategoryTheory.Linear R C
                                              K L M : CochainComplex C Int
                                              n✝ : Int
                                              γ✝ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n✝
                                              n' a : Int
                                              γ : CochainComplex.HomComplex.Cochain K ((CategoryTheory.shiftFunctor (Cochain …
                                              n : Int
                                              hn : Eq (HAdd.hAdd n' a) n
                                              p q : Int
                                              hpq : Eq (HAdd.hAdd p n) q
                                              ⊢ Eq q (HAdd.hAdd (HAdd.hAdd p n') a)
                                            -/
    (L.shiftFunctorObjXIso a (p + n') q (by rw [← hpq, add_assoc, hn])).hom)
                                            /-
                                              🎉 no goals
                                            -/


lemma rightUnshift_v {n' a : ℤ} (γ : Cochain K (L⟦a⟧) n') (n : ℤ) (hn : n' + a = n)
    (p q : ℤ) (hpq : p + n = q) (p' : ℤ) (hp' : p + n' = p') :
    (γ.rightUnshift n hn).v p q hpq = γ.v p p' hp' ≫
                                        /-
                                          C : Type u
                                          inst✝³ : CategoryTheory.Category.{v, u} C
                                          inst✝² : CategoryTheory.Preadditive C
                                          R : Type u_1
                                          inst✝¹ : Ring R
                                          inst✝ : CategoryTheory.Linear R C
                                          K L M : CochainComplex C Int
                                          n✝ : Int
                                          γ✝ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n✝
                                          n' a : Int
                                          γ : CochainComplex.HomComplex.Cochain K ((CategoryTheory.shiftFunctor (Cochain …
                                          n : Int
                                          hn : Eq (HAdd.hAdd n' a) n
                                          p q : Int
                                          hpq : Eq (HAdd.hAdd p n) q
                                          p' : Int
                                          hp' : Eq (HAdd.hAdd p n') p'
                                          ⊢ Eq q (HAdd.hAdd p' a)
                                        -/
      (L.shiftFunctorObjXIso a p' q (by rw [← hpq, ← hn, ← add_assoc, hp'])).hom := by
                                        /-
                                          🎉 no goals
                                        -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n' a : Int
    γ : CochainComplex.HomComplex.Cochain K ((CategoryTheory.shiftFunctor (Cochain …
    n : Int
    hn : Eq (HAdd.hAdd n' a) n
    p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    p' : Int
    hp' : Eq (HAdd.hAdd p n') p'
    ⊢ Eq ((γ.rightUnshift n hn).v p q hpq) (CategoryTheory.CategoryStruct.comp (γ. …
  -/
  subst hp'
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n' a : Int
    γ : CochainComplex.HomComplex.Cochain K ((CategoryTheory.shiftFunctor (Cochain …
    n : Int
    hn : Eq (HAdd.hAdd n' a) n
    p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    ⊢ Eq ((γ.rightUnshift n hn).v p q hpq) (CategoryTheory.CategoryStruct.comp (γ. …
  -/
  dsimp only [rightUnshift]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n' a : Int
    γ : CochainComplex.HomComplex.Cochain K ((CategoryTheory.shiftFunctor (Cochain …
    n : Int
    hn : Eq (HAdd.hAdd n' a) n
    p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.mk fun p q hpq => CategoryTheory.Cate …
  -/
  simp only [mk_v]
  /-
    🎉 no goals
  -/


/-- The map `Cochain (K⟦a⟧) L n' → Cochain K L n` when `n + a = n'`. -/
def leftUnshift {n' a : ℤ} (γ : Cochain (K⟦a⟧) L n') (n : ℤ) (hn : n + a = n') :
    Cochain K L n :=
  Cochain.mk (fun p q hpq => (a * n' + ((a * (a-1))/2)).negOnePow •
                                           /-
                                             C : Type u
                                             inst✝³ : CategoryTheory.Category.{v, u} C
                                             inst✝² : CategoryTheory.Preadditive C
                                             R : Type u_1
                                             inst✝¹ : Ring R
                                             inst✝ : CategoryTheory.Linear R C
                                             K L M : CochainComplex C Int
                                             n✝ : Int
                                             γ✝ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n✝
                                             n' a : Int
                                             γ : CochainComplex.HomComplex.Cochain ((CategoryTheory.shiftFunctor (CochainCo …
                                             n : Int
                                             hn : Eq (HAdd.hAdd n a) n'
                                             p q : Int
                                             hpq : Eq (HAdd.hAdd p n) q
                                             ⊢ Eq p (HAdd.hAdd (HSub.hSub p a) a)
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
    (K.shiftFunctorObjXIso a (p - a) p (by omega)).inv ≫ γ.v (p-a) q (by omega))
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


lemma leftUnshift_v {n' a : ℤ} (γ : Cochain (K⟦a⟧) L n') (n : ℤ) (hn : n + a = n')
    (p q : ℤ) (hpq : p + n = q) (p' : ℤ) (hp' : p' + n' = q) :
    (γ.leftUnshift n hn).v p q hpq = (a * n' + ((a * (a-1))/2)).negOnePow •
                                        /-
                                          C : Type u
                                          inst✝³ : CategoryTheory.Category.{v, u} C
                                          inst✝² : CategoryTheory.Preadditive C
                                          R : Type u_1
                                          inst✝¹ : Ring R
                                          inst✝ : CategoryTheory.Linear R C
                                          K L M : CochainComplex C Int
                                          n✝ : Int
                                          γ✝ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n✝
                                          n' a : Int
                                          γ : CochainComplex.HomComplex.Cochain ((CategoryTheory.shiftFunctor (CochainCo …
                                          n : Int
                                          hn : Eq (HAdd.hAdd n a) n'
                                          p q : Int
                                          hpq : Eq (HAdd.hAdd p n) q
                                          p' : Int
                                          hp' : Eq (HAdd.hAdd p' n') q
                                          ⊢ Eq p (HAdd.hAdd p' a)
                                        -/
                                        /-
                                          🎉 no goals
                                        -/
      (K.shiftFunctorObjXIso a p' p (by omega)).inv ≫ γ.v p' q (by omega) := by
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n' a : Int
    γ : CochainComplex.HomComplex.Cochain ((CategoryTheory.shiftFunctor (CochainCo …
    n : Int
    hn : Eq (HAdd.hAdd n a) n'
    p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    p' : Int
    hp' : Eq (HAdd.hAdd p' n') q
    ⊢ Eq ((γ.leftUnshift n hn).v p q hpq) (HSMul.hSMul (HAdd.hAdd (HMul.hMul a n') …
  -/
  obtain rfl : p' = p - a := by omega
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n' a : Int
    γ : CochainComplex.HomComplex.Cochain ((CategoryTheory.shiftFunctor (CochainCo …
    n : Int
    hn : Eq (HAdd.hAdd n a) n'
    p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    hp' : Eq (HAdd.hAdd (HSub.hSub p a) n') q
    ⊢ Eq ((γ.leftUnshift n hn).v p q hpq) (HSMul.hSMul (HAdd.hAdd (HMul.hMul a n') …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The map `Cochain K L n → Cochain (K⟦a⟧) (L⟦a⟧) n`. -/
def shift (a : ℤ) : Cochain (K⟦a⟧) (L⟦a⟧) n :=
  Cochain.mk (fun p q hpq => (K.shiftFunctorObjXIso a p _ rfl).hom ≫
                            /-
                              C : Type u
                              inst✝³ : CategoryTheory.Category.{v, u} C
                              inst✝² : CategoryTheory.Preadditive C
                              R : Type u_1
                              inst✝¹ : Ring R
                              inst✝ : CategoryTheory.Linear R C
                              K L M : CochainComplex C Int
                              n : Int
                              γ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n
                              a p q : Int
                              hpq : Eq (HAdd.hAdd p n) q
                              ⊢ Eq (HAdd.hAdd (HAdd.hAdd p a) n) (HAdd.hAdd q a)
                            -/
    γ.v (p + a) (q + a) (by omega) ≫ (L.shiftFunctorObjXIso a q _ rfl).inv)
                            /-
                              🎉 no goals
                            -/


lemma shift_v (a : ℤ) (p q : ℤ) (hpq : p + n = q) (p' q' : ℤ)
    (hp' : p' = p + a) (hq' : q' = q + a) :
    (γ.shift a).v p q hpq = (K.shiftFunctorObjXIso a p p' hp').hom ≫
                    /-
                      C : Type u
                      inst✝³ : CategoryTheory.Category.{v, u} C
                      inst✝² : CategoryTheory.Preadditive C
                      R : Type u_1
                      inst✝¹ : Ring R
                      inst✝ : CategoryTheory.Linear R C
                      K L M : CochainComplex C Int
                      n : Int
                      γ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n
                      a p q : Int
                      hpq : Eq (HAdd.hAdd p n) q
                      p' q' : Int
                      hp' : Eq p' (HAdd.hAdd p a)
                      hq' : Eq q' (HAdd.hAdd q a)
                      ⊢ Eq (HAdd.hAdd p' n) q'
                    -/
      γ.v p' q' (by rw [hp', hq', ← hpq, add_assoc, add_comm a, add_assoc]) ≫
                    /-
                      🎉 no goals
                    -/
      (L.shiftFunctorObjXIso a q q' hq').inv := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    p' q' : Int
    hp' : Eq p' (HAdd.hAdd p a)
    hq' : Eq q' (HAdd.hAdd q a)
    ⊢ Eq ((γ.shift a).v p q hpq) (CategoryTheory.CategoryStruct.comp (K.shiftFunct …
  -/
  subst hp' hq'
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    ⊢ Eq ((γ.shift a).v p q hpq) (CategoryTheory.CategoryStruct.comp (K.shiftFunct …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma shift_v' (a : ℤ) (p q : ℤ) (hpq : p + n = q) :
                                                    /-
                                                      C : Type u
                                                      inst✝³ : CategoryTheory.Category.{v, u} C
                                                      inst✝² : CategoryTheory.Preadditive C
                                                      R : Type u_1
                                                      inst✝¹ : Ring R
                                                      inst✝ : CategoryTheory.Linear R C
                                                      K L M : CochainComplex C Int
                                                      n : Int
                                                      γ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n
                                                      a p q : Int
                                                      hpq : Eq (HAdd.hAdd p n) q
                                                      ⊢ Eq (HAdd.hAdd (HAdd.hAdd p a) n) (HAdd.hAdd q a)
                                                    -/
    (γ.shift a).v p q hpq = γ.v (p + a) (q + a) (by omega) := by
                                                    /-
                                                      🎉 no goals
                                                    -/
  simp only [shift_v γ a p q hpq _ _ rfl rfl, shiftFunctor_obj_X, shiftFunctorObjXIso,
    HomologicalComplex.XIsoOfEq_rfl, Iso.refl_hom, Iso.refl_inv, comp_id, id_comp]


@[simp]
lemma rightUnshift_rightShift (a n' : ℤ) (hn' : n' + a = n) :
    (γ.rightShift a n' hn').rightUnshift n hn' = γ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    ⊢ Eq ((γ.rightShift a n' hn').rightUnshift n hn') γ
  -/
  ext p q hpq
  simp only [rightUnshift_v _ n hn' p q hpq (p + n') rfl,
    γ.rightShift_v _ _ hn' p (p + n') rfl q hpq,
    shiftFunctorObjXIso, assoc, Iso.inv_hom_id, comp_id]


@[simp]
lemma rightShift_rightUnshift {a n' : ℤ} (γ : Cochain K (L⟦a⟧) n') (n : ℤ) (hn' : n' + a = n) :
    (γ.rightUnshift n hn').rightShift a n' hn' = γ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    a n' : Int
    γ : CochainComplex.HomComplex.Cochain K ((CategoryTheory.shiftFunctor (Cochain …
    n : Int
    hn' : Eq (HAdd.hAdd n' a) n
    ⊢ Eq ((γ.rightUnshift n hn').rightShift a n' hn') γ
  -/
  ext p q hpq
  simp only [(γ.rightUnshift n hn').rightShift_v a n' hn' p q hpq (p + n) rfl,
    γ.rightUnshift_v n hn' p (p + n) rfl q hpq,
    shiftFunctorObjXIso, assoc, Iso.hom_inv_id, comp_id]


@[simp]
lemma leftUnshift_leftShift (a n' : ℤ) (hn' : n + a = n') :
    (γ.leftShift a n' hn').leftUnshift n hn' = γ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    ⊢ Eq ((γ.leftShift a n' hn').leftUnshift n hn') γ
  -/
  ext p q hpq
  rw [(γ.leftShift a n' hn').leftUnshift_v n hn' p q hpq (q-n') (by omega),
    γ.leftShift_v a n' hn' (q-n') q (by omega) p hpq, Linear.comp_units_smul,
    Iso.inv_hom_id_assoc, smul_smul, Int.units_mul_self, one_smul]


@[simp]
lemma leftShift_leftUnshift {a n' : ℤ} (γ : Cochain (K⟦a⟧) L n') (n : ℤ) (hn' : n + a = n') :
    (γ.leftUnshift n hn').leftShift a n' hn' = γ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    a n' : Int
    γ : CochainComplex.HomComplex.Cochain ((CategoryTheory.shiftFunctor (CochainCo …
    n : Int
    hn' : Eq (HAdd.hAdd n a) n'
    ⊢ Eq ((γ.leftUnshift n hn').leftShift a n' hn') γ
  -/
  ext p q hpq
  rw [(γ.leftUnshift n hn').leftShift_v a n' hn' p q hpq (q-n) (by omega),
    γ.leftUnshift_v n hn' (q-n) q (by omega) p hpq, Linear.comp_units_smul, smul_smul,
    Iso.hom_inv_id_assoc, Int.units_mul_self, one_smul]


@[simp]
lemma rightShift_add (a n' : ℤ) (hn' : n' + a = n) :
    (γ₁ + γ₂).rightShift a n' hn' = γ₁.rightShift a n' hn' + γ₂.rightShift a n' hn' := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    ⊢ Eq ((HAdd.hAdd γ₁ γ₂).rightShift a n' hn') (HAdd.hAdd (γ₁.rightShift a n' hn …
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    p q : Int
    hpq : Eq (HAdd.hAdd p n') q
    ⊢ Eq (((HAdd.hAdd γ₁ γ₂).rightShift a n' hn').v p q hpq) ((HAdd.hAdd (γ₁.right …
  -/
  dsimp
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    p q : Int
    hpq : Eq (HAdd.hAdd p n') q
    ⊢ Eq (((HAdd.hAdd γ₁ γ₂).rightShift a n' hn').v p q hpq) (HAdd.hAdd ((γ₁.right …
  -/
  simp only [rightShift_v _ a n' hn' p q hpq _ rfl, add_v, add_comp]
  /-
    🎉 no goals
  -/


@[simp]
lemma leftShift_add (a n' : ℤ) (hn' : n + a = n') :
    (γ₁ + γ₂).leftShift a n' hn' = γ₁.leftShift a n' hn' + γ₂.leftShift a n' hn' := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    ⊢ Eq ((HAdd.hAdd γ₁ γ₂).leftShift a n' hn') (HAdd.hAdd (γ₁.leftShift a n' hn') …
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    p q : Int
    hpq : Eq (HAdd.hAdd p n') q
    ⊢ Eq (((HAdd.hAdd γ₁ γ₂).leftShift a n' hn').v p q hpq) ((HAdd.hAdd (γ₁.leftSh …
  -/
  dsimp
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    p q : Int
    hpq : Eq (HAdd.hAdd p n') q
    ⊢ Eq (((HAdd.hAdd γ₁ γ₂).leftShift a n' hn').v p q hpq) (HAdd.hAdd ((γ₁.leftSh …
  -/
  simp only [leftShift_v _ a n' hn' p q hpq (p + a) (by omega), add_v, comp_add, smul_add]
  /-
    🎉 no goals
  -/


@[simp]
lemma shift_add (a : ℤ) :
    (γ₁ + γ₂).shift a = γ₁.shift a + γ₂.shift a := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n
    a : Int
    ⊢ Eq ((HAdd.hAdd γ₁ γ₂).shift a) (HAdd.hAdd (γ₁.shift a) (γ₂.shift a))
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n
    a p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    ⊢ Eq (((HAdd.hAdd γ₁ γ₂).shift a).v p q hpq) ((HAdd.hAdd (γ₁.shift a) (γ₂.shif …
  -/
  dsimp
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n
    a p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    ⊢ Eq (((HAdd.hAdd γ₁ γ₂).shift a).v p q hpq) (HAdd.hAdd ((γ₁.shift a).v p q hp …
  -/
  simp only [shift_v', add_v]
  /-
    🎉 no goals
  -/


/-- The additive equivalence `Cochain K L n ≃+ Cochain K L⟦a⟧ n'` when `n' + a = n`. -/
@[simps]
def rightShiftAddEquiv (n a n' : ℤ) (hn' : n' + a = n) :
    Cochain K L n ≃+ Cochain K (L⟦a⟧) n' where
  toFun γ := γ.rightShift a n' hn'
  invFun γ := γ.rightUnshift n hn'
                   /-
                     C : Type u
                     inst✝³ : CategoryTheory.Category.{v, u} C
                     inst✝² : CategoryTheory.Preadditive C
                     R : Type u_1
                     inst✝¹ : Ring R
                     inst✝ : CategoryTheory.Linear R C
                     K L M : CochainComplex C Int
                     n✝ : Int
                     γ✝ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n✝
                     n a n' : Int
                     hn' : Eq (HAdd.hAdd n' a) n
                     γ : CochainComplex.HomComplex.Cochain K L n
                     ⊢ Eq ((fun γ => γ.rightUnshift n hn') ((fun γ => γ.rightShift a n' hn') γ)) γ
                   -/
  left_inv γ := by dsimp; simp only [rightUnshift_rightShift]
                          /-
                            🎉 no goals
                          -/
                    /-
                      C : Type u
                      inst✝³ : CategoryTheory.Category.{v, u} C
                      inst✝² : CategoryTheory.Preadditive C
                      R : Type u_1
                      inst✝¹ : Ring R
                      inst✝ : CategoryTheory.Linear R C
                      K L M : CochainComplex C Int
                      n✝ : Int
                      γ✝ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n✝
                      n a n' : Int
                      hn' : Eq (HAdd.hAdd n' a) n
                      γ : CochainComplex.HomComplex.Cochain K ((CategoryTheory.shiftFunctor (Cochain …
                      ⊢ Eq ((fun γ => γ.rightShift a n' hn') ((fun γ => γ.rightUnshift n hn') γ)) γ
                    -/
  right_inv γ := by dsimp; simp only [rightShift_rightUnshift]
                           /-
                             🎉 no goals
                           -/
                      /-
                        C : Type u
                        inst✝³ : CategoryTheory.Category.{v, u} C
                        inst✝² : CategoryTheory.Preadditive C
                        R : Type u_1
                        inst✝¹ : Ring R
                        inst✝ : CategoryTheory.Linear R C
                        K L M : CochainComplex C Int
                        n✝ : Int
                        γ✝ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n✝
                        n a n' : Int
                        hn' : Eq (HAdd.hAdd n' a) n
                        γ γ' : CochainComplex.HomComplex.Cochain K L n
                        ⊢ Eq ({ toFun := fun γ => γ.rightShift a n' hn', invFun := fun γ => γ.rightUns …
                      -/
  map_add' γ γ' := by dsimp; simp only [rightShift_add]
                             /-
                               🎉 no goals
                             -/


/-- The additive equivalence `Cochain K L n ≃+ Cochain (K⟦a⟧) L n'` when `n + a = n'`. -/
@[simps]
def leftShiftAddEquiv (n a n' : ℤ) (hn' : n + a = n') :
    Cochain K L n ≃+ Cochain (K⟦a⟧) L n' where
  toFun γ := γ.leftShift a n' hn'
  invFun γ := γ.leftUnshift n hn'
                   /-
                     C : Type u
                     inst✝³ : CategoryTheory.Category.{v, u} C
                     inst✝² : CategoryTheory.Preadditive C
                     R : Type u_1
                     inst✝¹ : Ring R
                     inst✝ : CategoryTheory.Linear R C
                     K L M : CochainComplex C Int
                     n✝ : Int
                     γ✝ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n✝
                     n a n' : Int
                     hn' : Eq (HAdd.hAdd n a) n'
                     γ : CochainComplex.HomComplex.Cochain K L n
                     ⊢ Eq ((fun γ => γ.leftUnshift n hn') ((fun γ => γ.leftShift a n' hn') γ)) γ
                   -/
  left_inv γ := by dsimp; simp only [leftUnshift_leftShift]
                          /-
                            🎉 no goals
                          -/
                    /-
                      C : Type u
                      inst✝³ : CategoryTheory.Category.{v, u} C
                      inst✝² : CategoryTheory.Preadditive C
                      R : Type u_1
                      inst✝¹ : Ring R
                      inst✝ : CategoryTheory.Linear R C
                      K L M : CochainComplex C Int
                      n✝ : Int
                      γ✝ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n✝
                      n a n' : Int
                      hn' : Eq (HAdd.hAdd n a) n'
                      γ : CochainComplex.HomComplex.Cochain ((CategoryTheory.shiftFunctor (CochainCo …
                      ⊢ Eq ((fun γ => γ.leftShift a n' hn') ((fun γ => γ.leftUnshift n hn') γ)) γ
                    -/
  right_inv γ := by dsimp; simp only [leftShift_leftUnshift]
                           /-
                             🎉 no goals
                           -/
                      /-
                        C : Type u
                        inst✝³ : CategoryTheory.Category.{v, u} C
                        inst✝² : CategoryTheory.Preadditive C
                        R : Type u_1
                        inst✝¹ : Ring R
                        inst✝ : CategoryTheory.Linear R C
                        K L M : CochainComplex C Int
                        n✝ : Int
                        γ✝ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n✝
                        n a n' : Int
                        hn' : Eq (HAdd.hAdd n a) n'
                        γ γ' : CochainComplex.HomComplex.Cochain K L n
                        ⊢ Eq ({ toFun := fun γ => γ.leftShift a n' hn', invFun := fun γ => γ.leftUnshi …
                      -/
  map_add' γ γ' := by dsimp; simp only [leftShift_add]
                             /-
                               🎉 no goals
                             -/


/-- The additive map `Cochain K L n →+ Cochain (K⟦a⟧) (L⟦a⟧) n`. -/
@[simps!]
def shiftAddHom (n a : ℤ) : Cochain K L n →+ Cochain (K⟦a⟧) (L⟦a⟧) n :=
                                            /-
                                              C : Type u
                                              inst✝³ : CategoryTheory.Category.{v, u} C
                                              inst✝² : CategoryTheory.Preadditive C
                                              R : Type u_1
                                              inst✝¹ : Ring R
                                              inst✝ : CategoryTheory.Linear R C
                                              K L M : CochainComplex C Int
                                              n✝ : Int
                                              γ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n✝
                                              n a : Int
                                              ⊢ ∀ (a_1 b : CochainComplex.HomComplex.Cochain K L n), Eq ((fun γ => γ.shift a …
                                            -/
  AddMonoidHom.mk' (fun γ => γ.shift a) (by intros; dsimp; simp only [shift_add])
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
lemma rightShift_zero (a n' : ℤ) (hn' : n' + a = n) :
    (0 : Cochain K L n).rightShift a n' hn' = 0 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    ⊢ Eq (CochainComplex.HomComplex.Cochain.rightShift 0 a n' hn') 0
  -/
  change rightShiftAddEquiv K L n a n' hn' 0 = 0
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.rightShiftAddEquiv K L n a n' hn') 0) 0
  -/
  apply _root_.map_zero
  /-
    🎉 no goals
  -/


@[simp]
lemma rightUnshift_zero (a n' : ℤ) (hn' : n' + a = n) :
    (0 : Cochain K (L⟦a⟧) n').rightUnshift n hn' = 0 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    ⊢ Eq (CochainComplex.HomComplex.Cochain.rightUnshift 0 n hn') 0
  -/
  change (rightShiftAddEquiv K L n a n' hn').symm 0 = 0
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.rightShiftAddEquiv K L n a n' hn').sy …
  -/
  apply _root_.map_zero
  /-
    🎉 no goals
  -/


@[simp]
lemma leftShift_zero (a n' : ℤ) (hn' : n + a = n') :
    (0 : Cochain K L n).leftShift a n' hn' = 0 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    ⊢ Eq (CochainComplex.HomComplex.Cochain.leftShift 0 a n' hn') 0
  -/
  change leftShiftAddEquiv K L n a n' hn' 0 = 0
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.leftShiftAddEquiv K L n a n' hn') 0) 0
  -/
  apply _root_.map_zero
  /-
    🎉 no goals
  -/


@[simp]
lemma leftUnshift_zero (a n' : ℤ) (hn' : n + a = n') :
    (0 : Cochain (K⟦a⟧) L n').leftUnshift n hn' = 0 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    ⊢ Eq (CochainComplex.HomComplex.Cochain.leftUnshift 0 n hn') 0
  -/
  change (leftShiftAddEquiv K L n a n' hn').symm 0 = 0
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.leftShiftAddEquiv K L n a n' hn').sym …
  -/
  apply _root_.map_zero
  /-
    🎉 no goals
  -/


@[simp]
lemma shift_zero (a : ℤ) :
    (0 : Cochain K L n).shift a = 0 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n a : Int
    ⊢ Eq (CochainComplex.HomComplex.Cochain.shift 0 a) 0
  -/
  change shiftAddHom K L n a 0 = 0
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n a : Int
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.shiftAddHom K L n a) 0) 0
  -/
  apply _root_.map_zero
  /-
    🎉 no goals
  -/


@[simp]
lemma rightShift_neg (a n' : ℤ) (hn' : n' + a = n) :
    (-γ).rightShift a n' hn' = -γ.rightShift a n' hn' := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    ⊢ Eq ((Neg.neg γ).rightShift a n' hn') (Neg.neg (γ.rightShift a n' hn'))
  -/
  change rightShiftAddEquiv K L n a n' hn' (-γ) = _
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.rightShiftAddEquiv K L n a n' hn') (N …
  -/
  apply _root_.map_neg
  /-
    🎉 no goals
  -/


@[simp]
lemma rightUnshift_neg {n' a : ℤ} (γ : Cochain K (L⟦a⟧) n') (n : ℤ) (hn : n' + a = n) :
    (-γ).rightUnshift n hn = -γ.rightUnshift n hn := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n' a : Int
    γ : CochainComplex.HomComplex.Cochain K ((CategoryTheory.shiftFunctor (Cochain …
    n : Int
    hn : Eq (HAdd.hAdd n' a) n
    ⊢ Eq ((Neg.neg γ).rightUnshift n hn) (Neg.neg (γ.rightUnshift n hn))
  -/
  change (rightShiftAddEquiv K L n a n' hn).symm (-γ) = _
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n' a : Int
    γ : CochainComplex.HomComplex.Cochain K ((CategoryTheory.shiftFunctor (Cochain …
    n : Int
    hn : Eq (HAdd.hAdd n' a) n
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.rightShiftAddEquiv K L n a n' hn).sym …
  -/
  apply _root_.map_neg
  /-
    🎉 no goals
  -/


@[simp]
lemma leftShift_neg (a n' : ℤ) (hn' : n + a = n') :
    (-γ).leftShift a n' hn' = -γ.leftShift a n' hn' := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    ⊢ Eq ((Neg.neg γ).leftShift a n' hn') (Neg.neg (γ.leftShift a n' hn'))
  -/
  change leftShiftAddEquiv K L n a n' hn' (-γ) = _
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.leftShiftAddEquiv K L n a n' hn') (Ne …
  -/
  apply _root_.map_neg
  /-
    🎉 no goals
  -/


@[simp]
lemma leftUnshift_neg {n' a : ℤ} (γ : Cochain (K⟦a⟧) L n') (n : ℤ) (hn : n + a = n') :
    (-γ).leftUnshift n hn = -γ.leftUnshift n hn := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n' a : Int
    γ : CochainComplex.HomComplex.Cochain ((CategoryTheory.shiftFunctor (CochainCo …
    n : Int
    hn : Eq (HAdd.hAdd n a) n'
    ⊢ Eq ((Neg.neg γ).leftUnshift n hn) (Neg.neg (γ.leftUnshift n hn))
  -/
  change (leftShiftAddEquiv K L n a n' hn).symm (-γ) = _
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n' a : Int
    γ : CochainComplex.HomComplex.Cochain ((CategoryTheory.shiftFunctor (CochainCo …
    n : Int
    hn : Eq (HAdd.hAdd n a) n'
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.leftShiftAddEquiv K L n a n' hn).symm …
  -/
  apply _root_.map_neg
  /-
    🎉 no goals
  -/


@[simp]
lemma shift_neg (a : ℤ) :
    (-γ).shift a = -γ.shift a := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a : Int
    ⊢ Eq ((Neg.neg γ).shift a) (Neg.neg (γ.shift a))
  -/
  change shiftAddHom K L n a (-γ) = _
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a : Int
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.shiftAddHom K L n a) (Neg.neg γ)) (Ne …
  -/
  apply _root_.map_neg
  /-
    🎉 no goals
  -/


@[simp]
lemma rightUnshift_add {n' a : ℤ} (γ₁ γ₂ : Cochain K (L⟦a⟧) n') (n : ℤ) (hn : n' + a = n) :
    (γ₁ + γ₂).rightUnshift n hn = γ₁.rightUnshift n hn + γ₂.rightUnshift n hn := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n' a : Int
    γ₁ γ₂ : CochainComplex.HomComplex.Cochain K ((CategoryTheory.shiftFunctor (Coc …
    n : Int
    hn : Eq (HAdd.hAdd n' a) n
    ⊢ Eq ((HAdd.hAdd γ₁ γ₂).rightUnshift n hn) (HAdd.hAdd (γ₁.rightUnshift n hn) ( …
  -/
  change (rightShiftAddEquiv K L n a n' hn).symm (γ₁ + γ₂) = _
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n' a : Int
    γ₁ γ₂ : CochainComplex.HomComplex.Cochain K ((CategoryTheory.shiftFunctor (Coc …
    n : Int
    hn : Eq (HAdd.hAdd n' a) n
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.rightShiftAddEquiv K L n a n' hn).sym …
  -/
  apply _root_.map_add
  /-
    🎉 no goals
  -/


@[simp]
lemma leftUnshift_add {n' a : ℤ} (γ₁ γ₂ : Cochain (K⟦a⟧) L n') (n : ℤ) (hn : n + a = n') :
    (γ₁ + γ₂).leftUnshift n hn = γ₁.leftUnshift n hn + γ₂.leftUnshift n hn := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n' a : Int
    γ₁ γ₂ : CochainComplex.HomComplex.Cochain ((CategoryTheory.shiftFunctor (Cocha …
    n : Int
    hn : Eq (HAdd.hAdd n a) n'
    ⊢ Eq ((HAdd.hAdd γ₁ γ₂).leftUnshift n hn) (HAdd.hAdd (γ₁.leftUnshift n hn) (γ₂ …
  -/
  change (leftShiftAddEquiv K L n a n' hn).symm (γ₁ + γ₂) = _
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n' a : Int
    γ₁ γ₂ : CochainComplex.HomComplex.Cochain ((CategoryTheory.shiftFunctor (Cocha …
    n : Int
    hn : Eq (HAdd.hAdd n a) n'
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.leftShiftAddEquiv K L n a n' hn).symm …
  -/
  apply _root_.map_add
  /-
    🎉 no goals
  -/


@[simp]
lemma rightShift_smul (a n' : ℤ) (hn' : n' + a = n) (x : R) :
    (x • γ).rightShift a n' hn' = x • γ.rightShift a n' hn' := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    x : R
    ⊢ Eq ((HSMul.hSMul x γ).rightShift a n' hn') (HSMul.hSMul x (γ.rightShift a n' …
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    x : R
    p q : Int
    hpq : Eq (HAdd.hAdd p n') q
    ⊢ Eq (((HSMul.hSMul x γ).rightShift a n' hn').v p q hpq) ((HSMul.hSMul x (γ.ri …
  -/
  dsimp
  /-
    case h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    x : R
    p q : Int
    hpq : Eq (HAdd.hAdd p n') q
    ⊢ Eq (((HSMul.hSMul x γ).rightShift a n' hn').v p q hpq) (HSMul.hSMul x ((γ.ri …
  -/
  simp only [rightShift_v _ a n' hn' p q hpq _ rfl, smul_v, Linear.smul_comp]
  /-
    🎉 no goals
  -/


@[simp]
lemma leftShift_smul (a n' : ℤ) (hn' : n + a = n') (x : R) :
    (x • γ).leftShift a n' hn' = x • γ.leftShift a n' hn' := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    x : R
    ⊢ Eq ((HSMul.hSMul x γ).leftShift a n' hn') (HSMul.hSMul x (γ.leftShift a n' h …
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    x : R
    p q : Int
    hpq : Eq (HAdd.hAdd p n') q
    ⊢ Eq (((HSMul.hSMul x γ).leftShift a n' hn').v p q hpq) ((HSMul.hSMul x (γ.lef …
  -/
  dsimp
  simp only [leftShift_v _ a n' hn' p q hpq (p + a) (by omega), smul_v, Linear.comp_smul,
    smul_comm x]


@[simp]
lemma shift_smul (a : ℤ) (x : R) :
    (x • γ).shift a = x • (γ.shift a) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a : Int
    x : R
    ⊢ Eq ((HSMul.hSMul x γ).shift a) (HSMul.hSMul x (γ.shift a))
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a : Int
    x : R
    p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    ⊢ Eq (((HSMul.hSMul x γ).shift a).v p q hpq) ((HSMul.hSMul x (γ.shift a)).v p  …
  -/
  dsimp
  /-
    case h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a : Int
    x : R
    p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    ⊢ Eq (((HSMul.hSMul x γ).shift a).v p q hpq) (HSMul.hSMul x ((γ.shift a).v p q …
  -/
  simp only [shift_v', smul_v]
  /-
    🎉 no goals
  -/


/-- The linear equivalence `Cochain K L n ≃+ Cochain K L⟦a⟧ n'` when `n' + a = n` and
the category is `R`-linear. -/
@[simps!]
def rightShiftLinearEquiv (n a n' : ℤ) (hn' : n' + a = n) :
    Cochain K L n ≃ₗ[R] Cochain K (L⟦a⟧) n' :=
  (rightShiftAddEquiv K L n a n' hn').toLinearEquiv
                   /-
                     C : Type u
                     inst✝³ : CategoryTheory.Category.{v, u} C
                     inst✝² : CategoryTheory.Preadditive C
                     R : Type u_1
                     inst✝¹ : Ring R
                     inst✝ : CategoryTheory.Linear R C
                     K L M : CochainComplex C Int
                     n✝ : Int
                     γ✝ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n✝
                     n a n' : Int
                     hn' : Eq (HAdd.hAdd n' a) n
                     x : R
                     γ : CochainComplex.HomComplex.Cochain K L n
                     ⊢ Eq ((CochainComplex.HomComplex.Cochain.rightShiftAddEquiv K L n a n' hn') (H …
                   -/
    (fun x γ => by dsimp; simp only [rightShift_smul])
                          /-
                            🎉 no goals
                          -/


/-- The additive equivalence `Cochain K L n ≃+ Cochain (K⟦a⟧) L n'` when `n + a = n'` and
the category is `R`-linear. -/
@[simps!]
def leftShiftLinearEquiv (n a n' : ℤ) (hn : n + a = n') :
    Cochain K L n ≃ₗ[R] Cochain (K⟦a⟧) L n' :=
  (leftShiftAddEquiv K L n a n' hn).toLinearEquiv
                   /-
                     C : Type u
                     inst✝³ : CategoryTheory.Category.{v, u} C
                     inst✝² : CategoryTheory.Preadditive C
                     R : Type u_1
                     inst✝¹ : Ring R
                     inst✝ : CategoryTheory.Linear R C
                     K L M : CochainComplex C Int
                     n✝ : Int
                     γ✝ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n✝
                     n a n' : Int
                     hn : Eq (HAdd.hAdd n a) n'
                     x : R
                     γ : CochainComplex.HomComplex.Cochain K L n
                     ⊢ Eq ((CochainComplex.HomComplex.Cochain.leftShiftAddEquiv K L n a n' hn) (HSM …
                   -/
    (fun x γ => by dsimp; simp only [leftShift_smul])
                          /-
                            🎉 no goals
                          -/


/-- The linear map `Cochain K L n ≃+ Cochain (K⟦a⟧) (L⟦a⟧) n` when the category is `R`-linear. -/
@[simps!]
def shiftLinearMap (n a : ℤ) :
    Cochain K L n →ₗ[R] Cochain (K⟦a⟧) (L⟦a⟧) n where
  toAddHom := shiftAddHom K L n a
                      /-
                        C : Type u
                        inst✝³ : CategoryTheory.Category.{v, u} C
                        inst✝² : CategoryTheory.Preadditive C
                        R : Type u_1
                        inst✝¹ : Ring R
                        inst✝ : CategoryTheory.Linear R C
                        K L M : CochainComplex C Int
                        n✝ : Int
                        γ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n✝
                        n a : Int
                        x✝¹ : R
                        x✝ : CochainComplex.HomComplex.Cochain K L n
                        ⊢ Eq ((↑(CochainComplex.HomComplex.Cochain.shiftAddHom K L n a)).toFun (HSMul. …
                      -/
  map_smul' _ _ := by dsimp; simp only [shift_smul]
                             /-
                               🎉 no goals
                             -/


@[simp]
lemma rightShift_units_smul (a n' : ℤ) (hn' : n' + a = n) (x : Rˣ) :
    (x • γ).rightShift a n' hn' = x • γ.rightShift a n' hn' := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    x : Units R
    ⊢ Eq ((HSMul.hSMul x γ).rightShift a n' hn') (HSMul.hSMul x (γ.rightShift a n' …
  -/
  apply rightShift_smul
  /-
    🎉 no goals
  -/


@[simp]
lemma leftShift_units_smul (a n' : ℤ) (hn' : n + a = n') (x : Rˣ) :
    (x • γ).leftShift a n' hn' = x • γ.leftShift a n' hn' := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    x : Units R
    ⊢ Eq ((HSMul.hSMul x γ).leftShift a n' hn') (HSMul.hSMul x (γ.leftShift a n' h …
  -/
  apply leftShift_smul
  /-
    🎉 no goals
  -/


@[simp]
lemma shift_units_smul (a : ℤ) (x : Rˣ) :
    (x • γ).shift a = x • (γ.shift a) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a : Int
    x : Units R
    ⊢ Eq ((HSMul.hSMul x γ).shift a) (HSMul.hSMul x (γ.shift a))
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a : Int
    x : Units R
    p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    ⊢ Eq (((HSMul.hSMul x γ).shift a).v p q hpq) ((HSMul.hSMul x (γ.shift a)).v p  …
  -/
  dsimp
  /-
    case h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a : Int
    x : Units R
    p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    ⊢ Eq (((HSMul.hSMul x γ).shift a).v p q hpq) (HSMul.hSMul x ((γ.shift a).v p q …
  -/
  simp only [shift_v', units_smul_v]
  /-
    🎉 no goals
  -/


@[simp]
lemma rightUnshift_smul {n' a : ℤ} (γ : Cochain K (L⟦a⟧) n') (n : ℤ) (hn : n' + a = n) (x : R) :
    (x • γ).rightUnshift n hn = x • γ.rightUnshift n hn := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    K L : CochainComplex C Int
    n' a : Int
    γ : CochainComplex.HomComplex.Cochain K ((CategoryTheory.shiftFunctor (Cochain …
    n : Int
    hn : Eq (HAdd.hAdd n' a) n
    x : R
    ⊢ Eq ((HSMul.hSMul x γ).rightUnshift n hn) (HSMul.hSMul x (γ.rightUnshift n hn))
  -/
  change (rightShiftLinearEquiv  R K L n a n' hn).symm (x • γ) = _
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    K L : CochainComplex C Int
    n' a : Int
    γ : CochainComplex.HomComplex.Cochain K ((CategoryTheory.shiftFunctor (Cochain …
    n : Int
    hn : Eq (HAdd.hAdd n' a) n
    x : R
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.rightShiftLinearEquiv R K L n a n' hn …
  -/
  apply map_smul
  /-
    🎉 no goals
  -/


@[simp]
lemma rightUnshift_units_smul {n' a : ℤ} (γ : Cochain K (L⟦a⟧) n') (n : ℤ)
    (hn : n' + a = n) (x : Rˣ) :
    (x • γ).rightUnshift n hn = x • γ.rightUnshift n hn := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    K L : CochainComplex C Int
    n' a : Int
    γ : CochainComplex.HomComplex.Cochain K ((CategoryTheory.shiftFunctor (Cochain …
    n : Int
    hn : Eq (HAdd.hAdd n' a) n
    x : Units R
    ⊢ Eq ((HSMul.hSMul x γ).rightUnshift n hn) (HSMul.hSMul x (γ.rightUnshift n hn))
  -/
  apply rightUnshift_smul
  /-
    🎉 no goals
  -/


@[simp]
lemma leftUnshift_smul {n' a : ℤ} (γ : Cochain (K⟦a⟧) L n') (n : ℤ) (hn : n + a = n') (x : R) :
    (x • γ).leftUnshift n hn = x • γ.leftUnshift n hn := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    K L : CochainComplex C Int
    n' a : Int
    γ : CochainComplex.HomComplex.Cochain ((CategoryTheory.shiftFunctor (CochainCo …
    n : Int
    hn : Eq (HAdd.hAdd n a) n'
    x : R
    ⊢ Eq ((HSMul.hSMul x γ).leftUnshift n hn) (HSMul.hSMul x (γ.leftUnshift n hn))
  -/
  change (leftShiftLinearEquiv  R K L n a n' hn).symm (x • γ) = _
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    K L : CochainComplex C Int
    n' a : Int
    γ : CochainComplex.HomComplex.Cochain ((CategoryTheory.shiftFunctor (CochainCo …
    n : Int
    hn : Eq (HAdd.hAdd n a) n'
    x : R
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.leftShiftLinearEquiv R K L n a n' hn) …
  -/
  apply map_smul
  /-
    🎉 no goals
  -/


@[simp]
lemma leftUnshift_units_smul {n' a : ℤ} (γ : Cochain (K⟦a⟧) L n') (n : ℤ)
    (hn : n + a = n') (x : Rˣ) :
    (x • γ).leftUnshift n hn = x • γ.leftUnshift n hn := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    K L : CochainComplex C Int
    n' a : Int
    γ : CochainComplex.HomComplex.Cochain ((CategoryTheory.shiftFunctor (CochainCo …
    n : Int
    hn : Eq (HAdd.hAdd n a) n'
    x : Units R
    ⊢ Eq ((HSMul.hSMul x γ).leftUnshift n hn) (HSMul.hSMul x (γ.leftUnshift n hn))
  -/
  apply leftUnshift_smul
  /-
    🎉 no goals
  -/


lemma rightUnshift_comp {m : ℤ} {a : ℤ} (γ' : Cochain L (M⟦a⟧) m) {nm : ℤ} (hnm : n + m = nm)
    (nm' : ℤ) (hnm' : nm + a = nm') (m' : ℤ) (hm' : m + a = m') :
    (γ.comp γ' hnm).rightUnshift nm' hnm' =
                                          /-
                                            C : Type u
                                            inst✝³ : CategoryTheory.Category.{v, u} C
                                            inst✝² : CategoryTheory.Preadditive C
                                            R : Type u_1
                                            inst✝¹ : Ring R
                                            inst✝ : CategoryTheory.Linear R C
                                            K L M : CochainComplex C Int
                                            n : Int
                                            γ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n
                                            m a : Int
                                            γ' : CochainComplex.HomComplex.Cochain L ((CategoryTheory.shiftFunctor (Cochai …
                                            nm : Int
                                            hnm : Eq (HAdd.hAdd n m) nm
                                            nm' : Int
                                            hnm' : Eq (HAdd.hAdd nm a) nm'
                                            m' : Int
                                            hm' : Eq (HAdd.hAdd m a) m'
                                            ⊢ Eq (HAdd.hAdd n m') nm'
                                          -/
      γ.comp (γ'.rightUnshift m' hm') (by omega) := by
                                          /-
                                            🎉 no goals
                                          -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L M : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    m a : Int
    γ' : CochainComplex.HomComplex.Cochain L ((CategoryTheory.shiftFunctor (Cochai …
    nm : Int
    hnm : Eq (HAdd.hAdd n m) nm
    nm' : Int
    hnm' : Eq (HAdd.hAdd nm a) nm'
    m' : Int
    hm' : Eq (HAdd.hAdd m a) m'
    ⊢ Eq ((γ.comp γ' hnm).rightUnshift nm' hnm') (γ.comp (γ'.rightUnshift m' hm') ⋯)
  -/
  ext p q hpq
  rw [(γ.comp γ' hnm).rightUnshift_v nm' hnm' p q hpq (p + n + m) (by omega),
    γ.comp_v γ' hnm p (p + n) (p + n + m) rfl rfl,
    comp_v _ _ (show n + m' = nm' by omega) p (p + n) q (by omega) (by omega),
    γ'.rightUnshift_v m' hm' (p + n) q (by omega) (p + n + m) rfl, assoc]


lemma leftShift_comp (a n' : ℤ) (hn' : n + a = n') {m t t' : ℤ} (γ' : Cochain L M m)
    (h : n + m = t) (ht' : t + a = t') :
    (γ.comp γ' h).leftShift a t' ht' = (a * m).negOnePow • (γ.leftShift a n' hn').comp γ'
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            inst✝² : CategoryTheory.Preadditive C
            R : Type u_1
            inst✝¹ : Ring R
            inst✝ : CategoryTheory.Linear R C
            K L M : CochainComplex C Int
            n : Int
            γ γ₁ γ₂ : CochainComplex.HomComplex.Cochain K L n
            a n' : Int
            hn' : Eq (HAdd.hAdd n a) n'
            m t t' : Int
            γ' : CochainComplex.HomComplex.Cochain L M m
            h : Eq (HAdd.hAdd n m) t
            ht' : Eq (HAdd.hAdd t a) t'
            ⊢ Eq (HAdd.hAdd n' m) t'
          -/
      (by rw [← ht', ← h, ← hn', add_assoc, add_comm a, add_assoc]) := by
          /-
            🎉 no goals
          -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L M : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    m t t' : Int
    γ' : CochainComplex.HomComplex.Cochain L M m
    h : Eq (HAdd.hAdd n m) t
    ht' : Eq (HAdd.hAdd t a) t'
    ⊢ Eq ((γ.comp γ' h).leftShift a t' ht') (HSMul.hSMul (HMul.hMul a m).negOnePow …
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L M : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    m t t' : Int
    γ' : CochainComplex.HomComplex.Cochain L M m
    h : Eq (HAdd.hAdd n m) t
    ht' : Eq (HAdd.hAdd t a) t'
    p q : Int
    hpq : Eq (HAdd.hAdd p t') q
    ⊢ Eq (((γ.comp γ' h).leftShift a t' ht').v p q hpq) ((HSMul.hSMul (HMul.hMul a …
  -/
  have h' : n' + m = t' := by omega
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L M : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    m t t' : Int
    γ' : CochainComplex.HomComplex.Cochain L M m
    h : Eq (HAdd.hAdd n m) t
    ht' : Eq (HAdd.hAdd t a) t'
    p q : Int
    hpq : Eq (HAdd.hAdd p t') q
    h' : Eq (HAdd.hAdd n' m) t'
    ⊢ Eq (((γ.comp γ' h).leftShift a t' ht').v p q hpq) ((HSMul.hSMul (HMul.hMul a …
  -/
  dsimp
  simp only [Cochain.comp_v _ _ h' p (p + n') q rfl (by omega),
    γ.leftShift_v a n' hn' p (p + n') rfl (p + a) (by omega),
    (γ.comp γ' h).leftShift_v a t' (by omega) p q hpq (p + a) (by omega),
    smul_smul, Linear.units_smul_comp, assoc, Int.negOnePow_add, ← mul_assoc, ← h',
    comp_v _ _ h (p + a) (p + n') q (by omega) (by omega)]
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L M : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    m t t' : Int
    γ' : CochainComplex.HomComplex.Cochain L M m
    h : Eq (HAdd.hAdd n m) t
    ht' : Eq (HAdd.hAdd t a) t'
    p q : Int
    hpq : Eq (HAdd.hAdd p t') q
    h' : Eq (HAdd.hAdd n' m) t'
    ⊢ Eq (HSMul.hSMul (HMul.hMul (HMul.hMul a (HAdd.hAdd n' m)).negOnePow (HDiv.hD …
  -/
  congr 2
  /-
    case h.e_a.e_a
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L M : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    m t t' : Int
    γ' : CochainComplex.HomComplex.Cochain L M m
    h : Eq (HAdd.hAdd n m) t
    ht' : Eq (HAdd.hAdd t a) t'
    p q : Int
    hpq : Eq (HAdd.hAdd p t') q
    h' : Eq (HAdd.hAdd n' m) t'
    ⊢ Eq (HMul.hMul a (HAdd.hAdd n' m)).negOnePow (HMul.hMul (HMul.hMul a m).negOn …
  -/
  rw [add_comm n', mul_add, Int.negOnePow_add]
  /-
    🎉 no goals
  -/


@[simp]
lemma leftShift_comp_zero_cochain (a n' : ℤ) (hn' : n + a = n') (γ' : Cochain L M 0) :
    (γ.comp γ' (add_zero n)).leftShift a n' hn' =
      (γ.leftShift a n' hn').comp γ' (add_zero n') := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L M : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    γ' : CochainComplex.HomComplex.Cochain L M 0
    ⊢ Eq ((γ.comp γ' ⋯).leftShift a n' hn') ((γ.leftShift a n' hn').comp γ' ⋯)
  -/
  rw [leftShift_comp γ a n' hn' γ' (add_zero _) hn', mul_zero, Int.negOnePow_zero, one_smul]
  /-
    🎉 no goals
  -/


lemma δ_rightShift (a n' m' : ℤ) (hn' : n' + a = n) (m : ℤ) (hm' : m' + a = m) :
    δ n' m' (γ.rightShift a n' hn') = a.negOnePow • (δ n m γ).rightShift a m' hm' := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' m' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    m : Int
    hm' : Eq (HAdd.hAdd m' a) m
    ⊢ Eq (CochainComplex.HomComplex.δ n' m' (γ.rightShift a n' hn')) (HSMul.hSMul  …
  -/
  by_cases hnm : n + 1 = m
    /-
      case pos
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      K L : CochainComplex C Int
      n : Int
      γ : CochainComplex.HomComplex.Cochain K L n
      a n' m' : Int
      hn' : Eq (HAdd.hAdd n' a) n
      m : Int
      hm' : Eq (HAdd.hAdd m' a) m
      hnm : Eq (HAdd.hAdd n 1) m
      ⊢ Eq (CochainComplex.HomComplex.δ n' m' (γ.rightShift a n' hn')) (HSMul.hSMul  …
    -/
  · have hnm' : n' + 1 = m' := by omega
    /-
      case pos
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      K L : CochainComplex C Int
      n : Int
      γ : CochainComplex.HomComplex.Cochain K L n
      a n' m' : Int
      hn' : Eq (HAdd.hAdd n' a) n
      m : Int
      hm' : Eq (HAdd.hAdd m' a) m
      hnm : Eq (HAdd.hAdd n 1) m
      hnm' : Eq (HAdd.hAdd n' 1) m'
      ⊢ Eq (CochainComplex.HomComplex.δ n' m' (γ.rightShift a n' hn')) (HSMul.hSMul  …
    -/
    ext p q hpq
    /-
      case pos.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      K L : CochainComplex C Int
      n : Int
      γ : CochainComplex.HomComplex.Cochain K L n
      a n' m' : Int
      hn' : Eq (HAdd.hAdd n' a) n
      m : Int
      hm' : Eq (HAdd.hAdd m' a) m
      hnm : Eq (HAdd.hAdd n 1) m
      hnm' : Eq (HAdd.hAdd n' 1) m'
      p q : Int
      hpq : Eq (HAdd.hAdd p m') q
      ⊢ Eq ((CochainComplex.HomComplex.δ n' m' (γ.rightShift a n' hn')).v p q hpq) ( …
    -/
    dsimp
    rw [(δ n m γ).rightShift_v a m' hm' p q hpq _ rfl,
      δ_v n m hnm _ p (p+m) rfl (p+n) (p+1) (by omega) rfl,
      δ_v n' m' hnm' _ p q hpq (p+n') (p+1) (by omega) rfl,
      γ.rightShift_v a n' hn' p (p+n') rfl (p+n) rfl,
      γ.rightShift_v a n' hn' (p+1) q _ (p+m) (by omega)]
    simp only [shiftFunctorObjXIso, shiftFunctor_obj_d',
      Linear.comp_units_smul, assoc, HomologicalComplex.XIsoOfEq_inv_comp_d,
      add_comp, HomologicalComplex.d_comp_XIsoOfEq_inv, Linear.units_smul_comp, smul_add,
      add_right_inj, smul_smul]
    /-
      case pos.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      K L : CochainComplex C Int
      n : Int
      γ : CochainComplex.HomComplex.Cochain K L n
      a n' m' : Int
      hn' : Eq (HAdd.hAdd n' a) n
      m : Int
      hm' : Eq (HAdd.hAdd m' a) m
      hnm : Eq (HAdd.hAdd n 1) m
      hnm' : Eq (HAdd.hAdd n' 1) m'
      p q : Int
      hpq : Eq (HAdd.hAdd p m') q
      ⊢ Eq (HSMul.hSMul m'.negOnePow (CategoryTheory.CategoryStruct.comp (K.d p (HAd …
    -/
    congr 1
    simp only [← hm', add_comm m', Int.negOnePow_add, ← mul_assoc,
      Int.units_mul_self, one_mul]
    /-
      case neg
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      K L : CochainComplex C Int
      n : Int
      γ : CochainComplex.HomComplex.Cochain K L n
      a n' m' : Int
      hn' : Eq (HAdd.hAdd n' a) n
      m : Int
      hm' : Eq (HAdd.hAdd m' a) m
      hnm : Not (Eq (HAdd.hAdd n 1) m)
      ⊢ Eq (CochainComplex.HomComplex.δ n' m' (γ.rightShift a n' hn')) (HSMul.hSMul  …
    -/
  · have hnm' : ¬ n' + 1 = m' := fun _ => hnm (by omega)
    /-
      case neg
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      K L : CochainComplex C Int
      n : Int
      γ : CochainComplex.HomComplex.Cochain K L n
      a n' m' : Int
      hn' : Eq (HAdd.hAdd n' a) n
      m : Int
      hm' : Eq (HAdd.hAdd m' a) m
      hnm : Not (Eq (HAdd.hAdd n 1) m)
      hnm' : Not (Eq (HAdd.hAdd n' 1) m')
      ⊢ Eq (CochainComplex.HomComplex.δ n' m' (γ.rightShift a n' hn')) (HSMul.hSMul  …
    -/
    rw [δ_shape _ _ hnm', δ_shape _ _ hnm, rightShift_zero, smul_zero]
    /-
      🎉 no goals
    -/


lemma δ_rightUnshift {a n' : ℤ} (γ : Cochain K (L⟦a⟧) n') (n : ℤ) (hn : n' + a = n)
    (m m' : ℤ) (hm' : m' + a = m) :
    δ n m (γ.rightUnshift n hn) = a.negOnePow • (δ n' m' γ).rightUnshift m hm' := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    a n' : Int
    γ : CochainComplex.HomComplex.Cochain K ((CategoryTheory.shiftFunctor (Cochain …
    n : Int
    hn : Eq (HAdd.hAdd n' a) n
    m m' : Int
    hm' : Eq (HAdd.hAdd m' a) m
    ⊢ Eq (CochainComplex.HomComplex.δ n m (γ.rightUnshift n hn)) (HSMul.hSMul a.ne …
  -/
  obtain ⟨γ', rfl⟩ := (rightShiftAddEquiv K L n a n' hn).surjective γ
  /-
    case intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    a n' n : Int
    hn : Eq (HAdd.hAdd n' a) n
    m m' : Int
    hm' : Eq (HAdd.hAdd m' a) m
    γ' : CochainComplex.HomComplex.Cochain K L n
    ⊢ Eq (CochainComplex.HomComplex.δ n m (((CochainComplex.HomComplex.Cochain.rig …
  -/
  dsimp
  simp only [rightUnshift_rightShift, γ'.δ_rightShift a n' m' hn m hm', rightUnshift_units_smul,
    smul_smul, Int.units_mul_self, one_smul]


lemma δ_leftShift (a n' m' : ℤ) (hn' : n + a = n') (m : ℤ) (hm' : m + a = m') :
    δ n' m' (γ.leftShift a n' hn') = a.negOnePow • (δ n m γ).leftShift a m' hm' := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' m' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    m : Int
    hm' : Eq (HAdd.hAdd m a) m'
    ⊢ Eq (CochainComplex.HomComplex.δ n' m' (γ.leftShift a n' hn')) (HSMul.hSMul a …
  -/
  by_cases hnm : n + 1 = m
    /-
      case pos
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      K L : CochainComplex C Int
      n : Int
      γ : CochainComplex.HomComplex.Cochain K L n
      a n' m' : Int
      hn' : Eq (HAdd.hAdd n a) n'
      m : Int
      hm' : Eq (HAdd.hAdd m a) m'
      hnm : Eq (HAdd.hAdd n 1) m
      ⊢ Eq (CochainComplex.HomComplex.δ n' m' (γ.leftShift a n' hn')) (HSMul.hSMul a …
    -/
  · have hnm' : n' + 1 = m' := by omega
    /-
      case pos
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      K L : CochainComplex C Int
      n : Int
      γ : CochainComplex.HomComplex.Cochain K L n
      a n' m' : Int
      hn' : Eq (HAdd.hAdd n a) n'
      m : Int
      hm' : Eq (HAdd.hAdd m a) m'
      hnm : Eq (HAdd.hAdd n 1) m
      hnm' : Eq (HAdd.hAdd n' 1) m'
      ⊢ Eq (CochainComplex.HomComplex.δ n' m' (γ.leftShift a n' hn')) (HSMul.hSMul a …
    -/
    ext p q hpq
    /-
      case pos.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      K L : CochainComplex C Int
      n : Int
      γ : CochainComplex.HomComplex.Cochain K L n
      a n' m' : Int
      hn' : Eq (HAdd.hAdd n a) n'
      m : Int
      hm' : Eq (HAdd.hAdd m a) m'
      hnm : Eq (HAdd.hAdd n 1) m
      hnm' : Eq (HAdd.hAdd n' 1) m'
      p q : Int
      hpq : Eq (HAdd.hAdd p m') q
      ⊢ Eq ((CochainComplex.HomComplex.δ n' m' (γ.leftShift a n' hn')).v p q hpq) (( …
    -/
    dsimp
    rw [(δ n m γ).leftShift_v a m' hm' p q hpq (p+a) (by omega),
      δ_v n m hnm _ (p+a) q (by omega) (p+n') (p+1+a) (by omega) (by omega),
      δ_v n' m' hnm' _ p q hpq (p+n') (p+1) (by omega) rfl,
      γ.leftShift_v a n' hn' p (p+n') rfl (p+a) (by omega),
      γ.leftShift_v a n' hn' (p+1) q (by omega) (p+1+a) (by omega)]
    simp only [shiftFunctor_obj_X, shiftFunctorObjXIso, HomologicalComplex.XIsoOfEq_rfl,
      Iso.refl_hom, id_comp, Linear.units_smul_comp, shiftFunctor_obj_d',
      Linear.comp_units_smul, smul_add, smul_smul]
    /-
      case pos.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      K L : CochainComplex C Int
      n : Int
      γ : CochainComplex.HomComplex.Cochain K L n
      a n' m' : Int
      hn' : Eq (HAdd.hAdd n a) n'
      m : Int
      hm' : Eq (HAdd.hAdd m a) m'
      hnm : Eq (HAdd.hAdd n 1) m
      hnm' : Eq (HAdd.hAdd n' 1) m'
      p q : Int
      hpq : Eq (HAdd.hAdd p m') q
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (HMul.hMul a n') (HDiv.hDiv (HMul.hMul …
    -/
    congr 2
      /-
        case pos.h.e_a.e_a
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Preadditive C
        K L : CochainComplex C Int
        n : Int
        γ : CochainComplex.HomComplex.Cochain K L n
        a n' m' : Int
        hn' : Eq (HAdd.hAdd n a) n'
        m : Int
        hm' : Eq (HAdd.hAdd m a) m'
        hnm : Eq (HAdd.hAdd n 1) m
        hnm' : Eq (HAdd.hAdd n' 1) m'
        p q : Int
        hpq : Eq (HAdd.hAdd p m') q
        ⊢ Eq (HAdd.hAdd (HMul.hMul a n') (HDiv.hDiv (HMul.hMul a (HSub.hSub a 1)) 2)). …
      -/
    · rw [← hnm', add_comm n', mul_add, mul_one]
      /-
        case pos.h.e_a.e_a
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Preadditive C
        K L : CochainComplex C Int
        n : Int
        γ : CochainComplex.HomComplex.Cochain K L n
        a n' m' : Int
        hn' : Eq (HAdd.hAdd n a) n'
        m : Int
        hm' : Eq (HAdd.hAdd m a) m'
        hnm : Eq (HAdd.hAdd n 1) m
        hnm' : Eq (HAdd.hAdd n' 1) m'
        p q : Int
        hpq : Eq (HAdd.hAdd p m') q
        ⊢ Eq (HAdd.hAdd (HMul.hMul a n') (HDiv.hDiv (HMul.hMul a (HSub.hSub a 1)) 2)). …
      -/
      simp only [Int.negOnePow_add, ← mul_assoc, Int.units_mul_self, one_mul]
      /-
        🎉 no goals
      -/
      /-
        case pos.h.e_a.e_a
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Preadditive C
        K L : CochainComplex C Int
        n : Int
        γ : CochainComplex.HomComplex.Cochain K L n
        a n' m' : Int
        hn' : Eq (HAdd.hAdd n a) n'
        m : Int
        hm' : Eq (HAdd.hAdd m a) m'
        hnm : Eq (HAdd.hAdd n 1) m
        hnm' : Eq (HAdd.hAdd n' 1) m'
        p q : Int
        hpq : Eq (HAdd.hAdd p m') q
        ⊢ Eq (HMul.hMul m'.negOnePow (HMul.hMul (HAdd.hAdd (HMul.hMul a n') (HDiv.hDiv …
      -/
    · simp only [← Int.negOnePow_add, ← hn', ← hm', ← hnm]
      /-
        case pos.h.e_a.e_a
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Preadditive C
        K L : CochainComplex C Int
        n : Int
        γ : CochainComplex.HomComplex.Cochain K L n
        a n' m' : Int
        hn' : Eq (HAdd.hAdd n a) n'
        m : Int
        hm' : Eq (HAdd.hAdd m a) m'
        hnm : Eq (HAdd.hAdd n 1) m
        hnm' : Eq (HAdd.hAdd n' 1) m'
        p q : Int
        hpq : Eq (HAdd.hAdd p m') q
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) a) (HAdd.hAdd (HAdd.hAdd (HMul.hMul …
      -/
      congr 1
      /-
        case pos.h.e_a.e_a.e_n
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Preadditive C
        K L : CochainComplex C Int
        n : Int
        γ : CochainComplex.HomComplex.Cochain K L n
        a n' m' : Int
        hn' : Eq (HAdd.hAdd n a) n'
        m : Int
        hm' : Eq (HAdd.hAdd m a) m'
        hnm : Eq (HAdd.hAdd n 1) m
        hnm' : Eq (HAdd.hAdd n' 1) m'
        p q : Int
        hpq : Eq (HAdd.hAdd p m') q
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) a) (HAdd.hAdd (HAdd.hAdd (HMul.hMul …
      -/
      linarith
      /-
        🎉 no goals
      -/
    /-
      case neg
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      K L : CochainComplex C Int
      n : Int
      γ : CochainComplex.HomComplex.Cochain K L n
      a n' m' : Int
      hn' : Eq (HAdd.hAdd n a) n'
      m : Int
      hm' : Eq (HAdd.hAdd m a) m'
      hnm : Not (Eq (HAdd.hAdd n 1) m)
      ⊢ Eq (CochainComplex.HomComplex.δ n' m' (γ.leftShift a n' hn')) (HSMul.hSMul a …
    -/
  · have hnm' : ¬ n' + 1 = m' := fun _ => hnm (by omega)
    /-
      case neg
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      K L : CochainComplex C Int
      n : Int
      γ : CochainComplex.HomComplex.Cochain K L n
      a n' m' : Int
      hn' : Eq (HAdd.hAdd n a) n'
      m : Int
      hm' : Eq (HAdd.hAdd m a) m'
      hnm : Not (Eq (HAdd.hAdd n 1) m)
      hnm' : Not (Eq (HAdd.hAdd n' 1) m')
      ⊢ Eq (CochainComplex.HomComplex.δ n' m' (γ.leftShift a n' hn')) (HSMul.hSMul a …
    -/
    rw [δ_shape _ _ hnm', δ_shape _ _ hnm, leftShift_zero, smul_zero]
    /-
      🎉 no goals
    -/


lemma δ_leftUnshift {a n' : ℤ} (γ : Cochain (K⟦a⟧) L n') (n : ℤ) (hn : n + a = n')
    (m m' : ℤ) (hm' : m + a = m') :
    δ n m (γ.leftUnshift n hn) = a.negOnePow • (δ n' m' γ).leftUnshift m hm' := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    a n' : Int
    γ : CochainComplex.HomComplex.Cochain ((CategoryTheory.shiftFunctor (CochainCo …
    n : Int
    hn : Eq (HAdd.hAdd n a) n'
    m m' : Int
    hm' : Eq (HAdd.hAdd m a) m'
    ⊢ Eq (CochainComplex.HomComplex.δ n m (γ.leftUnshift n hn)) (HSMul.hSMul a.neg …
  -/
  obtain ⟨γ', rfl⟩ := (leftShiftAddEquiv K L n a n' hn).surjective γ
  /-
    case intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    a n' n : Int
    hn : Eq (HAdd.hAdd n a) n'
    m m' : Int
    hm' : Eq (HAdd.hAdd m a) m'
    γ' : CochainComplex.HomComplex.Cochain K L n
    ⊢ Eq (CochainComplex.HomComplex.δ n m (((CochainComplex.HomComplex.Cochain.lef …
  -/
  dsimp
  simp only [leftUnshift_leftShift, γ'.δ_leftShift a n' m' hn m hm', leftUnshift_units_smul,
    smul_smul, Int.units_mul_self, one_smul]


@[simp]
lemma δ_shift (a m : ℤ) :
    δ n m (γ.shift a) = a.negOnePow • (δ n m γ).shift a := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a m : Int
    ⊢ Eq (CochainComplex.HomComplex.δ n m (γ.shift a)) (HSMul.hSMul a.negOnePow (( …
  -/
  by_cases hnm : n + 1 = m
    /-
      case pos
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      K L : CochainComplex C Int
      n : Int
      γ : CochainComplex.HomComplex.Cochain K L n
      a m : Int
      hnm : Eq (HAdd.hAdd n 1) m
      ⊢ Eq (CochainComplex.HomComplex.δ n m (γ.shift a)) (HSMul.hSMul a.negOnePow (( …
    -/
  · ext p q hpq
    /-
      case pos.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      K L : CochainComplex C Int
      n : Int
      γ : CochainComplex.HomComplex.Cochain K L n
      a m : Int
      hnm : Eq (HAdd.hAdd n 1) m
      p q : Int
      hpq : Eq (HAdd.hAdd p m) q
      ⊢ Eq ((CochainComplex.HomComplex.δ n m (γ.shift a)).v p q hpq) ((HSMul.hSMul a …
    -/
    dsimp
    simp only [shift_v', sub_add_cancel, shiftFunctor_obj_d',
      δ_v n m hnm _ p q hpq (q - 1) (p + 1) rfl rfl,
      δ_v n m hnm _ (p + a) (q + a) (by omega) (q - 1 + a) (p + 1 + a)
        (by omega) (by omega),
      smul_add, Linear.units_smul_comp, Linear.comp_units_smul, add_right_inj]
    /-
      case pos.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      K L : CochainComplex C Int
      n : Int
      γ : CochainComplex.HomComplex.Cochain K L n
      a m : Int
      hnm : Eq (HAdd.hAdd n 1) m
      p q : Int
      hpq : Eq (HAdd.hAdd p m) q
      ⊢ Eq (HSMul.hSMul m.negOnePow (HSMul.hSMul a.negOnePow (CategoryTheory.Categor …
    -/
    rw [smul_comm]
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      K L : CochainComplex C Int
      n : Int
      γ : CochainComplex.HomComplex.Cochain K L n
      a m : Int
      hnm : Not (Eq (HAdd.hAdd n 1) m)
      ⊢ Eq (CochainComplex.HomComplex.δ n m (γ.shift a)) (HSMul.hSMul a.negOnePow (( …
    -/
  · rw [δ_shape _ _ hnm, δ_shape _ _ hnm, shift_zero, smul_zero]
    /-
      🎉 no goals
    -/


lemma leftShift_rightShift (a n' : ℤ) (hn' : n' + a = n) :
    (γ.rightShift a n' hn').leftShift a n hn' =
      (a * n + (a * (a - 1)) / 2).negOnePow • γ.shift a := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    ⊢ Eq ((γ.rightShift a n' hn').leftShift a n hn') (HSMul.hSMul (HAdd.hAdd (HMul …
  -/
  ext p q hpq
  simp only [leftShift_v _ a n hn' p q hpq (p + a) (by omega),
    rightShift_v _ a n' hn' (p + a) q (by omega) (q + a) (by omega), units_smul_v, shift_v']
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    ⊢ Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul a n) (HDiv.hDiv (HMul.hMul a (HSub.hSu …
  -/
  dsimp
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n' a) n
    p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    ⊢ Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul a n) (HDiv.hDiv (HMul.hMul a (HSub.hSu …
  -/
  rw [id_comp, comp_id]
  /-
    🎉 no goals
  -/


lemma rightShift_leftShift (a n' : ℤ) (hn' : n + a = n') :
    (γ.leftShift a n' hn').rightShift a n hn' =
      (a * n' + (a * (a - 1)) / 2).negOnePow • γ.shift a := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    ⊢ Eq ((γ.leftShift a n' hn').rightShift a n hn') (HSMul.hSMul (HAdd.hAdd (HMul …
  -/
  ext p q hpq
  simp only [rightShift_v _ a n hn' p q hpq (q + a) (by omega),
    leftShift_v _ a n' hn' p (q + a) (by omega) (p + a) (by omega), units_smul_v, shift_v']
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul (HAdd.hAdd (HMul.hMul a  …
  -/
  dsimp
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain K L n
    a n' : Int
    hn' : Eq (HAdd.hAdd n a) n'
    p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul (HAdd.hAdd (HMul.hMul a  …
  -/
  rw [id_comp, comp_id]
  /-
    🎉 no goals
  -/


/-- The left and right shift of cochains commute only up to a sign. -/
lemma leftShift_rightShift_eq_negOnePow_rightShift_leftShift
    (a n' n'' : ℤ) (hn' : n' + a = n) (hn'' : n + a = n'') :
    (γ.rightShift a n' hn').leftShift a n hn' =
      a.negOnePow • (γ.leftShift a n'' hn'').rightShift a n hn'' := by
  rw [leftShift_rightShift, rightShift_leftShift, smul_smul, ← hn'', add_comm n a, mul_add,
    Int.negOnePow_add, Int.negOnePow_add, Int.negOnePow_add, Int.negOnePow_mul_self,
    ← mul_assoc, ← mul_assoc, Int.units_mul_self, one_mul]


/-- The map `Cocycle K L n → Cocycle K (L⟦a⟧) n'` when `n' + a = n`. -/
@[simps!]
def rightShift (γ : Cocycle K L n) (a n' : ℤ) (hn' : n' + a = n) :
    Cocycle K (L⟦a⟧) n' :=
  Cocycle.mk (γ.1.rightShift a n' hn') _ rfl (by
    simp only [Cochain.δ_rightShift _ a n' (n' + 1) hn' (n + 1) (by omega),
      δ_eq_zero, Cochain.rightShift_zero, smul_zero])


/-- The map `Cocycle K (L⟦a⟧) n' → Cocycle K L n` when `n' + a = n`. -/
@[simps!]
def rightUnshift {n' a : ℤ} (γ : Cocycle K (L⟦a⟧) n') (n : ℤ) (hn : n' + a = n) :
    Cocycle K L n :=
  Cocycle.mk (γ.1.rightUnshift n hn) _ rfl (by
    rw [Cochain.δ_rightUnshift _ n hn (n + 1) (n + 1 - a) (by omega),
      δ_eq_zero, Cochain.rightUnshift_zero, smul_zero])


/-- The map `Cocycle K L n → Cocycle (K⟦a⟧) L n'` when `n + a = n'`. -/
@[simps!]
def leftShift (γ : Cocycle K L n) (a n' : ℤ) (hn' : n + a = n') :
    Cocycle (K⟦a⟧) L n' :=
  Cocycle.mk (γ.1.leftShift a n' hn') _ rfl (by
    simp only [Cochain.δ_leftShift _ a n' (n' + 1) hn' (n + 1) (by omega),
      δ_eq_zero, Cochain.leftShift_zero, smul_zero])


/-- The map `Cocycle (K⟦a⟧) L n' → Cocycle K L n` when `n + a = n'`. -/
@[simps!]
def leftUnshift {n' a : ℤ} (γ : Cocycle (K⟦a⟧) L n') (n : ℤ) (hn : n + a = n') :
    Cocycle K L n :=
  Cocycle.mk (γ.1.leftUnshift n hn) _ rfl (by
    rw [Cochain.δ_leftUnshift _ n hn (n + 1) (n + 1 + a) rfl,
      δ_eq_zero, Cochain.leftUnshift_zero, smul_zero])


/-- The map `Cocycle K L n → Cocycle (K⟦a⟧) (L⟦a⟧) n`. -/
@[simps!]
def shift (γ : Cocycle K L n) (a : ℤ) :
    Cocycle (K⟦a⟧) (L⟦a⟧) n :=
  Cocycle.mk (γ.1.shift a) _ rfl
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          R : Type u_1
          inst✝¹ : Ring R
          inst✝ : CategoryTheory.Linear R C
          K L M : CochainComplex C Int
          n : Int
          γ : CochainComplex.HomComplex.Cocycle K L n
          a : Int
          ⊢ Eq (CochainComplex.HomComplex.δ n (HAdd.hAdd n 1) ((↑γ).shift a)) 0
        -/
    (by simp only [Cochain.δ_shift, δ_eq_zero, Cochain.shift_zero, smul_zero])
        /-
          🎉 no goals
        -/


