include hk hk' in
lemma comp_d_eq_zero_iff ⦃W : C⦄ (φ : W ⟶ K.X j) :
    φ ≫ K.d j k = 0 ↔ φ ≫ (K.extendXIso e hj').inv ≫ (K.extend e).d j' k' = 0 := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c
    e : c.Embedding c'
    j k : ι
    j' k' : ι'
    hj' : Eq (e.f j) j'
    hk : Eq (c.next j) k
    hk' : Eq (c'.next j') k'
    W : C
    φ : Quiver.Hom W (K.X j)
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp φ (K.d j k)) 0) (Eq (CategoryThe …
  -/
  by_cases hjk : c.Rel j k
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c
      e : c.Embedding c'
      j k : ι
      j' k' : ι'
      hj' : Eq (e.f j) j'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      W : C
      φ : Quiver.Hom W (K.X j)
      hjk : c.Rel j k
      ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp φ (K.d j k)) 0) (Eq (CategoryThe …
    -/
  · have hk' : e.f k = k' := by rw [← hk', ← hj', c'.next_eq' (e.rel hjk)]
    rw [K.extend_d_eq e hj' hk', Iso.inv_hom_id_assoc,
      ← cancel_mono (K.extendXIso e hk').inv, zero_comp, assoc]
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c
      e : c.Embedding c'
      j k : ι
      j' k' : ι'
      hj' : Eq (e.f j) j'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      W : C
      φ : Quiver.Hom W (K.X j)
      hjk : Not (c.Rel j k)
      ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp φ (K.d j k)) 0) (Eq (CategoryThe …
    -/
  · simp only [K.shape _ _ hjk, comp_zero, true_iff]
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c
      e : c.Embedding c'
      j k : ι
      j' k' : ι'
      hj' : Eq (e.f j) j'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      W : C
      φ : Quiver.Hom W (K.X j)
      hjk : Not (c.Rel j k)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.CategoryStruct.comp …
    -/
    rw [K.extend_d_from_eq_zero e j' k' j hj', comp_zero, comp_zero]
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c
      e : c.Embedding c'
      j k : ι
      j' k' : ι'
      hj' : Eq (e.f j) j'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      W : C
      φ : Quiver.Hom W (K.X j)
      hjk : Not (c.Rel j k)
      ⊢ Not (c.Rel j (c.next j))
    -/
    rw [hk]
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c
      e : c.Embedding c'
      j k : ι
      j' k' : ι'
      hj' : Eq (e.f j) j'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      W : C
      φ : Quiver.Hom W (K.X j)
      hjk : Not (c.Rel j k)
      ⊢ Not (c.Rel j k)
    -/
    exact hjk
    /-
      🎉 no goals
    -/


include hi hi' in
lemma d_comp_eq_zero_iff ⦃W : C⦄ (φ : K.X j ⟶ W) :
    K.d i j ≫ φ = 0 ↔ (K.extend e).d i' j' ≫ (K.extendXIso e hj').hom ≫ φ = 0 := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c
    e : c.Embedding c'
    i j : ι
    i' j' : ι'
    hj' : Eq (e.f j) j'
    hi : Eq (c.prev j) i
    hi' : Eq (c'.prev j') i'
    W : C
    φ : Quiver.Hom (K.X j) W
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp (K.d i j) φ) 0) (Eq (CategoryThe …
  -/
  by_cases hij : c.Rel i j
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i j : ι
      i' j' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      W : C
      φ : Quiver.Hom (K.X j) W
      hij : c.Rel i j
      ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp (K.d i j) φ) 0) (Eq (CategoryThe …
    -/
  · have hi' : e.f i = i' := by rw [← hi', ← hj', c'.prev_eq' (e.rel hij)]
    rw [K.extend_d_eq e hi' hj', assoc, assoc, Iso.inv_hom_id_assoc,
      ← cancel_epi (K.extendXIso e hi').hom, comp_zero]
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i j : ι
      i' j' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      W : C
      φ : Quiver.Hom (K.X j) W
      hij : Not (c.Rel i j)
      ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp (K.d i j) φ) 0) (Eq (CategoryThe …
    -/
  · simp only [K.shape _ _ hij, zero_comp, true_iff]
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i j : ι
      i' j' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      W : C
      φ : Quiver.Hom (K.X j) W
      hij : Not (c.Rel i j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((K.extend e).d i' j') (CategoryTheor …
    -/
    rw [K.extend_d_to_eq_zero e i' j' j hj', zero_comp]
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i j : ι
      i' j' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      W : C
      φ : Quiver.Hom (K.X j) W
      hij : Not (c.Rel i j)
      ⊢ Not (c.Rel (c.prev j) j)
    -/
    rw [hi]
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i j : ι
      i' j' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      W : C
      φ : Quiver.Hom (K.X j) W
      hij : Not (c.Rel i j)
      ⊢ Not (c.Rel i j)
    -/
    exact hij
    /-
      🎉 no goals
    -/


/-- The kernel fork of `(K.extend e).d j' k'` that is deduced from a kernel
fork of `K.d j k `. -/
@[simp]
noncomputable def kernelFork : KernelFork ((K.extend e).d j' k') :=
  KernelFork.ofι (cone.ι ≫ (extendXIso K e hj').inv)
        /-
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          C : Type u_3
          inst✝² : CategoryTheory.Category.{?u.12127, u_3} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝ : CategoryTheory.Limits.HasZeroObject C
          K L M : HomologicalComplex C c
          φ : Quiver.Hom K L
          φ' : Quiver.Hom L M
          e : c.Embedding c'
          i j k : ι
          i' j' k' : ι'
          hj' : Eq (e.f j) j'
          hi : Eq (c.prev j) i
          hi' : Eq (c'.prev j') i'
          hk : Eq (c.next j) k
          hk' : Eq (c'.next j') k'
          cone : CategoryTheory.Limits.KernelFork (K.d j k)
          hcone : CategoryTheory.Limits.IsLimit cone
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
    (by rw [assoc, ← comp_d_eq_zero_iff K e hj' hk hk' cone.ι, cone.condition])
        /-
          🎉 no goals
        -/


/-- The limit kernel fork of `(K.extend e).d j' k'` that is deduced from a limit
kernel fork of `K.d j k `. -/
noncomputable def isLimitKernelFork : IsLimit (kernelFork K e hj' hk hk' cone) :=
  KernelFork.isLimitOfIsLimitOfIff hcone ((K.extend e).d j' k')
    (extendXIso K e hj').symm (comp_d_eq_zero_iff K e hj' hk hk')


include hi hi' hcone in
/-- Auxiliary lemma for `lift_d_comp_eq_zero_iff`. -/
lemma lift_d_comp_eq_zero_iff' ⦃W : C⦄ (f' : K.X i ⟶ cone.pt)
    (hf' : f' ≫ cone.ι = K.d i j)
    (f'' : (K.extend e).X i' ⟶ cone.pt)
    (hf'' : f'' ≫ cone.ι ≫ (extendXIso K e hj').inv = (K.extend e).d i' j')
    (φ : cone.pt ⟶ W) :
    f' ≫ φ = 0 ↔ f'' ≫ φ = 0 := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c
    e : c.Embedding c'
    i j k : ι
    i' j' : ι'
    hj' : Eq (e.f j) j'
    hi : Eq (c.prev j) i
    hi' : Eq (c'.prev j') i'
    cone : CategoryTheory.Limits.KernelFork (K.d j k)
    hcone : CategoryTheory.Limits.IsLimit cone
    W : C
    f' : Quiver.Hom (K.X i) cone.pt
    hf' : Eq (CategoryTheory.CategoryStruct.comp f' (CategoryTheory.Limits.Fork.ι  …
    f'' : Quiver.Hom ((K.extend e).X i') cone.pt
    hf'' : Eq (CategoryTheory.CategoryStruct.comp f'' (CategoryTheory.CategoryStru …
    φ : Quiver.Hom cone.pt W
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f' φ) 0) (Eq (CategoryTheory.Cat …
  -/
  by_cases hij : c.Rel i j
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i j k : ι
      i' j' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      cone : CategoryTheory.Limits.KernelFork (K.d j k)
      hcone : CategoryTheory.Limits.IsLimit cone
      W : C
      f' : Quiver.Hom (K.X i) cone.pt
      hf' : Eq (CategoryTheory.CategoryStruct.comp f' (CategoryTheory.Limits.Fork.ι  …
      f'' : Quiver.Hom ((K.extend e).X i') cone.pt
      hf'' : Eq (CategoryTheory.CategoryStruct.comp f'' (CategoryTheory.CategoryStru …
      φ : Quiver.Hom cone.pt W
      hij : c.Rel i j
      ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f' φ) 0) (Eq (CategoryTheory.Cat …
    -/
  · have hi'' : e.f i = i' := by rw [← hi', ← hj', c'.prev_eq' (e.rel hij)]
    have : (K.extendXIso e hi'').hom ≫ f' = f'' := by
      apply Fork.IsLimit.hom_ext hcone
      rw [assoc, hf', ← cancel_mono (extendXIso K e hj').inv, assoc, assoc, hf'',
        K.extend_d_eq e hi'' hj']
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i j k : ι
      i' j' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      cone : CategoryTheory.Limits.KernelFork (K.d j k)
      hcone : CategoryTheory.Limits.IsLimit cone
      W : C
      f' : Quiver.Hom (K.X i) cone.pt
      hf' : Eq (CategoryTheory.CategoryStruct.comp f' (CategoryTheory.Limits.Fork.ι  …
      f'' : Quiver.Hom ((K.extend e).X i') cone.pt
      hf'' : Eq (CategoryTheory.CategoryStruct.comp f'' (CategoryTheory.CategoryStru …
      φ : Quiver.Hom cone.pt W
      hij : c.Rel i j
      hi'' : Eq (e.f i) i'
      this : Eq (CategoryTheory.CategoryStruct.comp (K.extendXIso e hi'').hom f') f''
      ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f' φ) 0) (Eq (CategoryTheory.Cat …
    -/
    rw [← cancel_epi (K.extendXIso e hi'').hom, comp_zero, ← this, assoc]
    /-
      🎉 no goals
    -/
  · have h₁ : f' = 0 := by
      apply Fork.IsLimit.hom_ext hcone
      simp only [zero_comp, hf', K.shape _ _ hij]
    have h₂ : f'' = 0 := by
      apply Fork.IsLimit.hom_ext hcone
      dsimp
      rw [← cancel_mono (extendXIso K e hj').inv, assoc, hf'', zero_comp, zero_comp,
        K.extend_d_to_eq_zero e i' j' j hj']
      rw [hi]
      exact hij
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i j k : ι
      i' j' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      cone : CategoryTheory.Limits.KernelFork (K.d j k)
      hcone : CategoryTheory.Limits.IsLimit cone
      W : C
      f' : Quiver.Hom (K.X i) cone.pt
      hf' : Eq (CategoryTheory.CategoryStruct.comp f' (CategoryTheory.Limits.Fork.ι  …
      f'' : Quiver.Hom ((K.extend e).X i') cone.pt
      hf'' : Eq (CategoryTheory.CategoryStruct.comp f'' (CategoryTheory.CategoryStru …
      φ : Quiver.Hom cone.pt W
      hij : Not (c.Rel i j)
      h₁ : Eq f' 0
      h₂ : Eq f'' 0
      ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f' φ) 0) (Eq (CategoryTheory.Cat …
    -/
    simp [h₁, h₂]
    /-
      🎉 no goals
    -/


include hi hi' in
lemma lift_d_comp_eq_zero_iff ⦃W : C⦄ (φ : cone.pt ⟶ W) :
    hcone.lift (KernelFork.ofι (K.d i j) (K.d_comp_d i j k)) ≫ φ = 0 ↔
      ((isLimitKernelFork K e hj' hk hk' cone hcone).lift
      (KernelFork.ofι ((K.extend e).d i' j') (d_comp_d _ _ _ _))) ≫ φ = 0 :=
  lift_d_comp_eq_zero_iff' K e hj' hi hi' cone hcone _ (hcone.fac _ _) _
    (IsLimit.fac _ _ WalkingParallelPair.zero) _


/-- Auxiliary definition for `extend.leftHomologyData`. -/
noncomputable def cokernelCofork :
    CokernelCofork ((isLimitKernelFork K e hj' hk hk' cone hcone).lift
      (KernelFork.ofι ((K.extend e).d i' j') (d_comp_d _ _ _ _))) :=
  CokernelCofork.ofπ cocone.π (by
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.32801, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i j k : ι
      i' j' k' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      cone : CategoryTheory.Limits.KernelFork (K.d j k)
      hcone : CategoryTheory.Limits.IsLimit cone
      cocone : CategoryTheory.Limits.CokernelCofork (hcone.lift (CategoryTheory.Limi …
      hcocone : CategoryTheory.Limits.IsColimit cocone
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomologicalComplex.extend.leftHomol …
    -/
    rw [← lift_d_comp_eq_zero_iff K e hj' hi hi' hk hk' cone hcone]
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.32801, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i j k : ι
      i' j' k' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      cone : CategoryTheory.Limits.KernelFork (K.d j k)
      hcone : CategoryTheory.Limits.IsLimit cone
      cocone : CategoryTheory.Limits.CokernelCofork (hcone.lift (CategoryTheory.Limi …
      hcocone : CategoryTheory.Limits.IsColimit cocone
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (hcone.lift (CategoryTheory.Limits.Ke …
    -/
    exact cocone.condition)
    /-
      🎉 no goals
    -/


/-- Auxiliary definition for `extend.leftHomologyData`. -/
noncomputable def isColimitCokernelCofork :
    IsColimit (cokernelCofork K e hj' hi hi' hk hk' cone hcone cocone) :=
  CokernelCofork.isColimitOfIsColimitOfIff' hcocone _
    (lift_d_comp_eq_zero_iff K e hj' hi hi' hk hk' cone hcone)


open leftHomologyData in
/-- The left homology data of `(K.extend e).sc' i' j' k'` that is deduced
from a left homology data of `K.sc' i j k`. -/
@[simps]
noncomputable def leftHomologyData (h : (K.sc' i j k).LeftHomologyData) :
    ((K.extend e).sc' i' j' k').LeftHomologyData where
  K := h.K
  H := h.H
  i := h.i ≫ (extendXIso K e hj').inv
  π := h.π
  wi := by
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.38238, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i j k : ι
      i' j' k' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      h : (K.sc' i j k).LeftHomologyData
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp h …
    -/
    dsimp
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.38238, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i j k : ι
      i' j' k' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      h : (K.sc' i j k).LeftHomologyData
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp h …
    -/
    rw [assoc, ← comp_d_eq_zero_iff K e hj' hk hk']
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.38238, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i j k : ι
      i' j' k' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      h : (K.sc' i j k).LeftHomologyData
      ⊢ Eq (CategoryTheory.CategoryStruct.comp h.i (K.d j k)) 0
    -/
    exact h.wi
    /-
      🎉 no goals
    -/
  hi := isLimitKernelFork K e hj' hk hk' _ h.hi
  wπ := by
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.38238, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i j k : ι
      i' j' k' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      h : (K.sc' i j k).LeftHomologyData
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomologicalComplex.extend.leftHomol …
    -/
    dsimp
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.38238, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i j k : ι
      i' j' k' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      h : (K.sc' i j k).LeftHomologyData
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomologicalComplex.extend.leftHomol …
    -/
    rw [← lift_d_comp_eq_zero_iff K e hj' hi hi' hk hk' _ h.hi]
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.38238, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i j k : ι
      i' j' k' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      h : (K.sc' i j k).LeftHomologyData
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.hi.lift (CategoryTheory.Limits.Ker …
    -/
    exact h.wπ
    /-
      🎉 no goals
    -/
  hπ := isColimitCokernelCofork K e hj' hi hi' hk hk' _ h.hi _ h.hπ


/-- The cokernel cofork of `(K.extend e).d i' j'` that is deduced from a cokernel
cofork of `K.d i j`. -/
@[simp]
noncomputable def cokernelCofork : CokernelCofork ((K.extend e).d i' j') :=
  CokernelCofork.ofπ ((extendXIso K e hj').hom ≫ cocone.π) (by
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.51868, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i j k : ι
      i' j' k' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      cocone : CategoryTheory.Limits.CokernelCofork (K.d i j)
      hcocone : CategoryTheory.Limits.IsColimit cocone
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((K.extend e).d i' j') (CategoryTheor …
    -/
    rw [← d_comp_eq_zero_iff K e hj' hi hi' cocone.π, cocone.condition])
    /-
      🎉 no goals
    -/


/-- The colimit cokernel cofork of `(K.extend e).d i' j'` that is deduced from a
colimit cokernel cofork of `K.d i j`. -/
noncomputable def isColimitCokernelCofork : IsColimit (cokernelCofork K e hj' hi hi' cocone) :=
  CokernelCofork.isColimitOfIsColimitOfIff hcocone ((K.extend e).d i' j')
    (extendXIso K e hj') (d_comp_eq_zero_iff K e hj' hi hi')


include hk hk' hcocone in
lemma d_comp_desc_eq_zero_iff' ⦃W : C⦄ (f' : cocone.pt ⟶ K.X k)
    (hf' : cocone.π ≫ f' = K.d j k)
    (f'' : cocone.pt ⟶ (K.extend e).X k')
    (hf'' : (extendXIso K e hj').hom ≫ cocone.π ≫ f'' = (K.extend e).d j' k')
    (φ : W ⟶ cocone.pt) :
    φ ≫ f' = 0 ↔ φ ≫ f'' = 0 := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c
    e : c.Embedding c'
    i j k : ι
    j' k' : ι'
    hj' : Eq (e.f j) j'
    hk : Eq (c.next j) k
    hk' : Eq (c'.next j') k'
    cocone : CategoryTheory.Limits.CokernelCofork (K.d i j)
    hcocone : CategoryTheory.Limits.IsColimit cocone
    W : C
    f' : Quiver.Hom cocone.pt (K.X k)
    hf' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π c …
    f'' : Quiver.Hom cocone.pt ((K.extend e).X k')
    hf'' : Eq (CategoryTheory.CategoryStruct.comp (K.extendXIso e hj').hom (Catego …
    φ : Quiver.Hom W cocone.pt
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp φ f') 0) (Eq (CategoryTheory.Cat …
  -/
  by_cases hjk : c.Rel j k
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i j k : ι
      j' k' : ι'
      hj' : Eq (e.f j) j'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      cocone : CategoryTheory.Limits.CokernelCofork (K.d i j)
      hcocone : CategoryTheory.Limits.IsColimit cocone
      W : C
      f' : Quiver.Hom cocone.pt (K.X k)
      hf' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π c …
      f'' : Quiver.Hom cocone.pt ((K.extend e).X k')
      hf'' : Eq (CategoryTheory.CategoryStruct.comp (K.extendXIso e hj').hom (Catego …
      φ : Quiver.Hom W cocone.pt
      hjk : c.Rel j k
      ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp φ f') 0) (Eq (CategoryTheory.Cat …
    -/
  · have hk'' : e.f k = k' := by rw [← hk', ← hj', c'.next_eq' (e.rel hjk)]
    have : f' ≫ (K.extendXIso e hk'').inv = f'' := by
      apply Cofork.IsColimit.hom_ext hcocone
      rw [reassoc_of% hf', ← cancel_epi (extendXIso K e hj').hom, hf'',
        K.extend_d_eq e hj' hk'']
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i j k : ι
      j' k' : ι'
      hj' : Eq (e.f j) j'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      cocone : CategoryTheory.Limits.CokernelCofork (K.d i j)
      hcocone : CategoryTheory.Limits.IsColimit cocone
      W : C
      f' : Quiver.Hom cocone.pt (K.X k)
      hf' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π c …
      f'' : Quiver.Hom cocone.pt ((K.extend e).X k')
      hf'' : Eq (CategoryTheory.CategoryStruct.comp (K.extendXIso e hj').hom (Catego …
      φ : Quiver.Hom W cocone.pt
      hjk : c.Rel j k
      hk'' : Eq (e.f k) k'
      this : Eq (CategoryTheory.CategoryStruct.comp f' (K.extendXIso e hk'').inv) f''
      ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp φ f') 0) (Eq (CategoryTheory.Cat …
    -/
    rw [← cancel_mono (K.extendXIso e hk'').inv, zero_comp, assoc, this]
    /-
      🎉 no goals
    -/
  · have h₁ : f' = 0 := by
      apply Cofork.IsColimit.hom_ext hcocone
      simp only [hf', comp_zero, K.shape _ _ hjk]
    have h₂ : f'' = 0 := by
      apply Cofork.IsColimit.hom_ext hcocone
      rw [← cancel_epi (extendXIso K e hj').hom, hf'', comp_zero, comp_zero,
        K.extend_d_from_eq_zero e j' k' j hj']
      rw [hk]
      exact hjk
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i j k : ι
      j' k' : ι'
      hj' : Eq (e.f j) j'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      cocone : CategoryTheory.Limits.CokernelCofork (K.d i j)
      hcocone : CategoryTheory.Limits.IsColimit cocone
      W : C
      f' : Quiver.Hom cocone.pt (K.X k)
      hf' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π c …
      f'' : Quiver.Hom cocone.pt ((K.extend e).X k')
      hf'' : Eq (CategoryTheory.CategoryStruct.comp (K.extendXIso e hj').hom (Catego …
      φ : Quiver.Hom W cocone.pt
      hjk : Not (c.Rel j k)
      h₁ : Eq f' 0
      h₂ : Eq f'' 0
      ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp φ f') 0) (Eq (CategoryTheory.Cat …
    -/
    simp [h₁, h₂]
    /-
      🎉 no goals
    -/


include hk hk' in
lemma d_comp_desc_eq_zero_iff ⦃W : C⦄ (φ : W ⟶ cocone.pt) :
    φ ≫ hcocone.desc (CokernelCofork.ofπ (K.d j k) (K.d_comp_d i j k)) = 0 ↔
      φ ≫ ((isColimitCokernelCofork K e hj' hi hi' cocone hcocone).desc
      (CokernelCofork.ofπ ((K.extend e).d j' k') (d_comp_d _ _ _ _))) = 0 :=
  d_comp_desc_eq_zero_iff' K e hj' hk hk' cocone hcocone _ (hcocone.fac _ _) _ (by
    simpa using (isColimitCokernelCofork K e hj' hi hi' cocone hcocone).fac _
      WalkingParallelPair.one) _


/-- Auxiliary definition for `extend.rightHomologyData`. -/
noncomputable def kernelFork :
    KernelFork ((isColimitCokernelCofork K e hj' hi hi' cocone hcocone).desc
      (CokernelCofork.ofπ ((K.extend e).d j' k') (d_comp_d _ _ _ _))) :=
  KernelFork.ofι cone.ι (by
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.71572, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i j k : ι
      i' j' k' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      cocone : CategoryTheory.Limits.CokernelCofork (K.d i j)
      hcocone : CategoryTheory.Limits.IsColimit cocone
      cone : CategoryTheory.Limits.KernelFork (hcocone.desc (CategoryTheory.Limits.C …
      hcone : CategoryTheory.Limits.IsLimit cone
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fork.ι cone) ( …
    -/
    rw [← d_comp_desc_eq_zero_iff K e hj' hi hi' hk hk' cocone hcocone]
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.71572, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i j k : ι
      i' j' k' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      cocone : CategoryTheory.Limits.CokernelCofork (K.d i j)
      hcocone : CategoryTheory.Limits.IsColimit cocone
      cone : CategoryTheory.Limits.KernelFork (hcocone.desc (CategoryTheory.Limits.C …
      hcone : CategoryTheory.Limits.IsLimit cone
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fork.ι cone) ( …
    -/
    exact cone.condition)
    /-
      🎉 no goals
    -/


/-- Auxiliary definition for `extend.rightHomologyData`. -/
noncomputable def isLimitKernelFork :
    IsLimit (kernelFork K e hj' hi hi' hk hk' cocone hcocone cone) :=
  KernelFork.isLimitOfIsLimitOfIff' hcone _
    (d_comp_desc_eq_zero_iff K e hj' hi hi' hk hk' cocone hcocone)


open rightHomologyData in
/-- The right homology data of `(K.extend e).sc' i' j' k'` that is deduced
from a right homology data of `K.sc' i j k`. -/
@[simps]
noncomputable def rightHomologyData (h : (K.sc' i j k).RightHomologyData) :
    ((K.extend e).sc' i' j' k').RightHomologyData where
  Q := h.Q
  H := h.H
  p := (extendXIso K e hj').hom ≫ h.p
  ι := h.ι
  wp := by
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.77070, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i j k : ι
      i' j' k' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      h : (K.sc' i j k).RightHomologyData
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((K.extend e).sc' i' j' k').f (Catego …
    -/
    dsimp
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.77070, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i j k : ι
      i' j' k' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      h : (K.sc' i j k).RightHomologyData
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((K.extend e).d i' j') (CategoryTheor …
    -/
    rw [← d_comp_eq_zero_iff K e hj' hi hi']
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.77070, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i j k : ι
      i' j' k' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      h : (K.sc' i j k).RightHomologyData
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d i j) h.p) 0
    -/
    exact h.wp
    /-
      🎉 no goals
    -/
  hp := isColimitCokernelCofork K e hj' hi hi' _ h.hp
  wι := by
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.77070, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i j k : ι
      i' j' k' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      h : (K.sc' i j k).RightHomologyData
      ⊢ Eq (CategoryTheory.CategoryStruct.comp h.ι ((HomologicalComplex.extend.right …
    -/
    dsimp
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.77070, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i j k : ι
      i' j' k' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      h : (K.sc' i j k).RightHomologyData
      ⊢ Eq (CategoryTheory.CategoryStruct.comp h.ι ((HomologicalComplex.extend.right …
    -/
    rw [← d_comp_desc_eq_zero_iff K e hj' hi hi' hk hk' _ h.hp]
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.77070, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i j k : ι
      i' j' k' : ι'
      hj' : Eq (e.f j) j'
      hi : Eq (c.prev j) i
      hi' : Eq (c'.prev j') i'
      hk : Eq (c.next j) k
      hk' : Eq (c'.next j') k'
      h : (K.sc' i j k).RightHomologyData
      ⊢ Eq (CategoryTheory.CategoryStruct.comp h.ι (h.hp.desc (CategoryTheory.Limits …
    -/
    exact h.wι
    /-
      🎉 no goals
    -/
  hι := isLimitKernelFork K e hj' hi hi' hk hk' _ h.hp _ h.hι


/-- The homology data of `(K.extend e).sc' i' j' k'` that is deduced
from a homology data of `K.sc' i j k`. -/
@[simps]
noncomputable def homologyData (h : (K.sc' i j k).HomologyData) :
    ((K.extend e).sc' i' j' k').HomologyData where
  left := leftHomologyData K e hj' hi hi' hk hk' h.left
  right := rightHomologyData K e hj' hi hi' hk hk' h.right
  iso := h.iso


/-- The homology data of `(K.extend e).sc j'` that is deduced
from a homology data of `K.sc' i j k`. -/
@[simps!]
noncomputable def homologyData' (h : (K.sc' i j k).HomologyData) :
    ((K.extend e).sc j').HomologyData :=
  homologyData K e hj' hi rfl hk rfl h


lemma hasHomology {j : ι} {j' : ι'} (hj' : e.f j = j') [K.HasHomology j] :
    (K.extend e).HasHomology j' :=
  ShortComplex.HasHomology.mk'
    (homologyData' K e hj' rfl rfl ((K.sc j).homologyData))


instance (j : ι) [K.HasHomology j] : (K.extend e).HasHomology (e.f j) :=
  hasHomology K e rfl


instance [∀ j, K.HasHomology j] (j' : ι') : (K.extend e).HasHomology j' := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K L M : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    e : c.Embedding c'
    inst✝ : ∀ (j : ι), K.HasHomology j
    j' : ι'
    ⊢ (K.extend e).HasHomology j'
  -/
  by_cases h : ∃ j, e.f j = j'
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_3} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      inst✝ : ∀ (j : ι), K.HasHomology j
      j' : ι'
      h : Exists fun j => Eq (e.f j) j'
      ⊢ (K.extend e).HasHomology j'
    -/
  · obtain ⟨j, rfl⟩ := h
    /-
      case pos.intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_3} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      inst✝ : ∀ (j : ι), K.HasHomology j
      j : ι
      ⊢ (K.extend e).HasHomology (e.f j)
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_3} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      inst✝ : ∀ (j : ι), K.HasHomology j
      j' : ι'
      h : Not (Exists fun j => Eq (e.f j) j')
      ⊢ (K.extend e).HasHomology j'
    -/
  · have hj := isZero_extend_X K e j' (by tauto)
    exact ShortComplex.HasHomology.mk'
      (ShortComplex.HomologyData.ofZeros _ (hj.eq_of_tgt _ _) (hj.eq_of_src _ _))


