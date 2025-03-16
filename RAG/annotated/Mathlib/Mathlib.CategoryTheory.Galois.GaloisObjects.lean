noncomputable instance {G : Type v} [Group G] [Finite G] :
    PreservesColimitsOfShape (SingleObj G) FintypeCat.incl.{w} := by
  /-
    G : Type v
    inst✝¹ : Group G
    inst✝ : Finite G
    ⊢ CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.SingleObj G)  …
  -/
  choose G' hg hf e using Finite.exists_type_univ_nonempty_mulEquiv G
  /-
    G : Type v
    inst✝¹ : Group G
    inst✝ : Finite G
    G' : Type ?u.590
    hg : Group G'
    hf : Fintype G'
    e : Nonempty (MulEquiv G G')
    ⊢ CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.SingleObj G)  …
  -/
  exact Limits.preservesColimitsOfShape_of_equiv (Classical.choice e).toSingleObjEquiv.symm _
  /-
    🎉 no goals
  -/


/-- A connected object `X` of `C` is Galois if the quotient `X / Aut X` is terminal. -/
class IsGalois {C : Type u₁} [Category.{u₂, u₁} C] [GaloisCategory C] (X : C)
    extends IsConnected X : Prop where
  quotientByAutTerminal : Nonempty (IsTerminal <| colimit <| SingleObj.functor <| Aut.toEnd X)


/-- The natural action of `Aut X` on `F.obj X`. -/
instance autMulFiber (F : C ⥤ FintypeCat.{w}) (X : C) : MulAction (Aut X) (F.obj X) where
  smul σ a := F.map σ.hom a
  one_smul a := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      X : C
      a : ↑(F.obj X)
      ⊢ Eq (HSMul.hSMul 1 a) a
    -/
    show F.map (𝟙 X) a = a
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      X : C
      a : ↑(F.obj X)
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.id X) a) a
    -/
    simp only [map_id, FintypeCat.id_apply]
    /-
      🎉 no goals
    -/
  mul_smul g h a := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      X : C
      g h : CategoryTheory.Aut X
      a : ↑(F.obj X)
      ⊢ Eq (HSMul.hSMul (HMul.hMul g h) a) (HSMul.hSMul g (HSMul.hSMul h a))
    -/
    show F.map (h.hom ≫ g.hom) a = (F.map h.hom ≫ F.map g.hom) a
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      X : C
      g h : CategoryTheory.Aut X
      a : ↑(F.obj X)
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp h.hom g.hom) a) (CategoryTheor …
    -/
    simp only [map_comp, FintypeCat.comp_apply]
    /-
      🎉 no goals
    -/


/-- For a connected object `X` of `C`, the quotient `X / Aut X` is terminal if and only if
the quotient `F.obj X / Aut X` has exactly one element. -/
noncomputable def quotientByAutTerminalEquivUniqueQuotient
    (X : C) [IsConnected X] :
    IsTerminal (colimit <| SingleObj.functor <| Aut.toEnd X) ≃
    Unique (MulAction.orbitRel.Quotient (Aut X) (F.obj X)) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    ⊢ Equiv (CategoryTheory.Limits.IsTerminal (CategoryTheory.Limits.colimit (Cate …
  -/
  let J : SingleObj (Aut X) ⥤ C := SingleObj.functor (Aut.toEnd X)
  let e : (F ⋙ FintypeCat.incl).obj (colimit J) ≅ _ :=
    preservesColimitIso (F ⋙ FintypeCat.incl) J ≪≫
    (Equiv.toIso <| SingleObj.Types.colimitEquivQuotient (J ⋙ F ⋙ FintypeCat.incl))
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    J : CategoryTheory.Functor (CategoryTheory.SingleObj (CategoryTheory.Aut X)) C …
    e : CategoryTheory.Iso ((F.comp FintypeCat.incl).obj (CategoryTheory.Limits.co …
    ⊢ Equiv (CategoryTheory.Limits.IsTerminal (CategoryTheory.Limits.colimit (Cate …
  -/
  apply Equiv.trans
  · apply (IsTerminal.isTerminalIffObj (F ⋙ FintypeCat.incl) _).trans
      (isLimitEmptyConeEquiv _ (asEmptyCone _) (asEmptyCone _) e)
  /-
    case e₂
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    J : CategoryTheory.Functor (CategoryTheory.SingleObj (CategoryTheory.Aut X)) C …
    e : CategoryTheory.Iso ((F.comp FintypeCat.incl).obj (CategoryTheory.Limits.co …
    ⊢ Equiv (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.asEmptyCone (Quo …
  -/
  exact Types.isTerminalEquivUnique _
  /-
    🎉 no goals
  -/


lemma isGalois_iff_aux (X : C) [IsConnected X] :
    IsGalois X ↔ Nonempty (IsTerminal <| colimit <| SingleObj.functor <| Aut.toEnd X) :=
  ⟨fun h ↦ h.quotientByAutTerminal, fun h ↦ ⟨h⟩⟩


/-- Given a fiber functor `F` and a connected object `X` of `C`. Then `X` is Galois if and only if
the natural action of `Aut X` on `F.obj X` is transitive. -/
theorem isGalois_iff_pretransitive (X : C) [IsConnected X] :
    IsGalois X ↔ MulAction.IsPretransitive (Aut X) (F.obj X) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    ⊢ Iff (CategoryTheory.PreGaloisCategory.IsGalois X) (MulAction.IsPretransitive …
  -/
  rw [isGalois_iff_aux, Equiv.nonempty_congr <| quotientByAutTerminalEquivUniqueQuotient F X]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    ⊢ Iff (Nonempty (Unique (MulAction.orbitRel.Quotient (CategoryTheory.Aut X) ↑( …
  -/
  exact (MulAction.pretransitive_iff_unique_quotient_of_nonempty (Aut X) (F.obj X)).symm
  /-
    🎉 no goals
  -/


/-- If `X` is Galois, the quotient `X / Aut X` is terminal. -/
noncomputable def isTerminalQuotientOfIsGalois (X : C) [IsGalois X] :
    IsTerminal <| colimit <| SingleObj.functor <| Aut.toEnd X :=
  Nonempty.some IsGalois.quotientByAutTerminal


/-- If `X` is Galois, then the action of `Aut X` on `F.obj X` is
transitive for every fiber functor `F`. -/
instance isPretransitive_of_isGalois (X : C) [IsGalois X] :
    MulAction.IsPretransitive (Aut X) (F.obj X) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois X
    ⊢ MulAction.IsPretransitive (CategoryTheory.Aut X) ↑(F.obj X)
  -/
  rw [← isGalois_iff_pretransitive]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois X
    ⊢ CategoryTheory.PreGaloisCategory.IsGalois X
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma stabilizer_normal_of_isGalois (X : C) [IsGalois X] (x : F.obj X) :
    Subgroup.Normal (MulAction.stabilizer (Aut F) x) where
  conj_mem n ninstab g := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X : C
      inst✝ : CategoryTheory.PreGaloisCategory.IsGalois X
      x : ↑(F.obj X)
      n : CategoryTheory.Aut F
      ninstab : Membership.mem (MulAction.stabilizer (CategoryTheory.Aut F) x) n
      g : CategoryTheory.Aut F
      ⊢ Membership.mem (MulAction.stabilizer (CategoryTheory.Aut F) x) (HMul.hMul (H …
    -/
    rw [MulAction.mem_stabilizer_iff]
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X : C
      inst✝ : CategoryTheory.PreGaloisCategory.IsGalois X
      x : ↑(F.obj X)
      n : CategoryTheory.Aut F
      ninstab : Membership.mem (MulAction.stabilizer (CategoryTheory.Aut F) x) n
      g : CategoryTheory.Aut F
      ⊢ Eq (HSMul.hSMul (HMul.hMul (HMul.hMul g n) (Inv.inv g)) x) x
    -/
    show g • n • (g⁻¹ • x) = x
    have : ∃ (φ : Aut X), F.map φ.hom x = g⁻¹ • x :=
      MulAction.IsPretransitive.exists_smul_eq x (g⁻¹ • x)
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X : C
      inst✝ : CategoryTheory.PreGaloisCategory.IsGalois X
      x : ↑(F.obj X)
      n : CategoryTheory.Aut F
      ninstab : Membership.mem (MulAction.stabilizer (CategoryTheory.Aut F) x) n
      g : CategoryTheory.Aut F
      this : Exists fun φ => Eq (F.map φ.hom x) (HSMul.hSMul (Inv.inv g) x)
      ⊢ Eq (HSMul.hSMul g (HSMul.hSMul n (HSMul.hSMul (Inv.inv g) x))) x
    -/
    obtain ⟨φ, h⟩ := this
    /-
      case intro
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X : C
      inst✝ : CategoryTheory.PreGaloisCategory.IsGalois X
      x : ↑(F.obj X)
      n : CategoryTheory.Aut F
      ninstab : Membership.mem (MulAction.stabilizer (CategoryTheory.Aut F) x) n
      g : CategoryTheory.Aut F
      φ : CategoryTheory.Aut X
      h : Eq (F.map φ.hom x) (HSMul.hSMul (Inv.inv g) x)
      ⊢ Eq (HSMul.hSMul g (HSMul.hSMul n (HSMul.hSMul (Inv.inv g) x))) x
    -/
    rw [← h, mulAction_naturality, ninstab, h]
    /-
      case intro
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X : C
      inst✝ : CategoryTheory.PreGaloisCategory.IsGalois X
      x : ↑(F.obj X)
      n : CategoryTheory.Aut F
      ninstab : Membership.mem (MulAction.stabilizer (CategoryTheory.Aut F) x) n
      g : CategoryTheory.Aut F
      φ : CategoryTheory.Aut X
      h : Eq (F.map φ.hom x) (HSMul.hSMul (Inv.inv g) x)
      ⊢ Eq (HSMul.hSMul g (HSMul.hSMul (Inv.inv g) x)) x
    -/
    simp
    /-
      🎉 no goals
    -/


theorem evaluation_aut_surjective_of_isGalois (A : C) [IsGalois A] (a : F.obj A) :
    Function.Surjective (fun f : Aut A ↦ F.map f.hom a) :=
  MulAction.IsPretransitive.exists_smul_eq a


theorem evaluation_aut_bijective_of_isGalois (A : C) [IsGalois A] (a : F.obj A) :
    Function.Bijective (fun f : Aut A ↦ F.map f.hom a) :=
  ⟨evaluation_aut_injective_of_isConnected F A a, evaluation_aut_surjective_of_isGalois F A a⟩


/-- For Galois `A` and a point `a` of the fiber of `A`, the evaluation at `A` as an equivalence. -/
noncomputable def evaluationEquivOfIsGalois (A : C) [IsGalois A] (a : F.obj A) : Aut A ≃ F.obj A :=
  Equiv.ofBijective _ (evaluation_aut_bijective_of_isGalois F A a)


@[simp]
lemma evaluationEquivOfIsGalois_apply (A : C) [IsGalois A] (a : F.obj A) (φ : Aut A) :
    evaluationEquivOfIsGalois F A a φ = F.map φ.hom a :=
  rfl


@[simp]
lemma evaluationEquivOfIsGalois_symm_fiber (A : C) [IsGalois A] (a b : F.obj A) :
    F.map ((evaluationEquivOfIsGalois F A a).symm b).hom a = b := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    A : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois A
    a b : ↑(F.obj A)
    ⊢ Eq (F.map ((CategoryTheory.PreGaloisCategory.evaluationEquivOfIsGalois F A a …
  -/
  change (evaluationEquivOfIsGalois F A a) _ = _
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    A : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois A
    a b : ↑(F.obj A)
    ⊢ Eq ((CategoryTheory.PreGaloisCategory.evaluationEquivOfIsGalois F A a) ((Cat …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- For a morphism from a connected object `A` to a Galois object `B` and an automorphism
of `A`, there exists a unique automorphism of `B` making the canonical diagram commute. -/
lemma exists_autMap {A B : C} (f : A ⟶ B) [IsConnected A] [IsGalois B] (σ : Aut A) :
    ∃! (τ : Aut B), f ≫ τ.hom = σ.hom ≫ f := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    A B : C
    f : Quiver.Hom A B
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
    σ : CategoryTheory.Aut A
    ⊢ ExistsUnique fun τ => Eq (CategoryTheory.CategoryStruct.comp f τ.hom) (Categ …
  -/
  let F := GaloisCategory.getFiberFunctor C
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    A B : C
    f : Quiver.Hom A B
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
    σ : CategoryTheory.Aut A
    F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
    ⊢ ExistsUnique fun τ => Eq (CategoryTheory.CategoryStruct.comp f τ.hom) (Categ …
  -/
  obtain ⟨a⟩ := nonempty_fiber_of_isConnected F A
  /-
    case intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    A B : C
    f : Quiver.Hom A B
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
    σ : CategoryTheory.Aut A
    F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
    a : ↑(F.obj A)
    ⊢ ExistsUnique fun τ => Eq (CategoryTheory.CategoryStruct.comp f τ.hom) (Categ …
  -/
  refine ⟨?_, ?_, ?_⟩
    /-
      case intro.refine_1
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      A B : C
      f : Quiver.Hom A B
      inst✝¹ : CategoryTheory.PreGaloisCategory.IsConnected A
      inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
      σ : CategoryTheory.Aut A
      F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
      a : ↑(F.obj A)
      ⊢ CategoryTheory.Aut B
    -/
  · exact (evaluationEquivOfIsGalois F B (F.map f a)).symm (F.map (σ.hom ≫ f) a)
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      A B : C
      f : Quiver.Hom A B
      inst✝¹ : CategoryTheory.PreGaloisCategory.IsConnected A
      inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
      σ : CategoryTheory.Aut A
      F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
      a : ↑(F.obj A)
      ⊢ (fun τ => Eq (CategoryTheory.CategoryStruct.comp f τ.hom) (CategoryTheory.Ca …
    -/
  · apply evaluation_injective_of_isConnected F A B a
    /-
      case intro.refine_2.a
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      A B : C
      f : Quiver.Hom A B
      inst✝¹ : CategoryTheory.PreGaloisCategory.IsConnected A
      inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
      σ : CategoryTheory.Aut A
      F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
      a : ↑(F.obj A)
      ⊢ Eq ((fun f => F.map f a) (CategoryTheory.CategoryStruct.comp f ((CategoryThe …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_3
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      A B : C
      f : Quiver.Hom A B
      inst✝¹ : CategoryTheory.PreGaloisCategory.IsConnected A
      inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
      σ : CategoryTheory.Aut A
      F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
      a : ↑(F.obj A)
      ⊢ ∀ (y : CategoryTheory.Aut B), (fun τ => Eq (CategoryTheory.CategoryStruct.co …
    -/
  · intro τ hτ
    /-
      case intro.refine_3
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      A B : C
      f : Quiver.Hom A B
      inst✝¹ : CategoryTheory.PreGaloisCategory.IsConnected A
      inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
      σ : CategoryTheory.Aut A
      F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
      a : ↑(F.obj A)
      τ : CategoryTheory.Aut B
      hτ : Eq (CategoryTheory.CategoryStruct.comp f τ.hom) (CategoryTheory.CategoryS …
      ⊢ Eq τ ((CategoryTheory.PreGaloisCategory.evaluationEquivOfIsGalois F B (F.map …
    -/
    apply evaluation_aut_injective_of_isConnected F B (F.map f a)
    /-
      case intro.refine_3.a
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      A B : C
      f : Quiver.Hom A B
      inst✝¹ : CategoryTheory.PreGaloisCategory.IsConnected A
      inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
      σ : CategoryTheory.Aut A
      F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
      a : ↑(F.obj A)
      τ : CategoryTheory.Aut B
      hτ : Eq (CategoryTheory.CategoryStruct.comp f τ.hom) (CategoryTheory.CategoryS …
      ⊢ Eq ((fun f_1 => F.map f_1.hom (F.map f a)) τ) ((fun f_1 => F.map f_1.hom (F. …
    -/
    simpa using congr_fun (F.congr_map hτ) a
    /-
      🎉 no goals
    -/


/-- A morphism from a connected object to a Galois object induces a map on automorphism
groups. This is a group homomorphism (see `autMapHom`). -/
noncomputable def autMap {A B : C} [IsConnected A] [IsGalois B] (f : A ⟶ B) (σ : Aut A) :
    Aut B :=
  (exists_autMap f σ).choose


@[simp]
lemma comp_autMap {A B : C} [IsConnected A] [IsGalois B] (f : A ⟶ B) (σ : Aut A) :
    f ≫ (autMap f σ).hom = σ.hom ≫ f :=
  (exists_autMap f σ).choose_spec.left


@[simp]
lemma comp_autMap_apply (F : C ⥤ FintypeCat.{w}) {A B : C} [IsConnected A] [IsGalois B]
    (f : A ⟶ B) (σ : Aut A) (a : F.obj A) :
    F.map (autMap f σ).hom (F.map f a) = F.map f (F.map σ.hom a) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    A B : C
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
    f : Quiver.Hom A B
    σ : CategoryTheory.Aut A
    a : ↑(F.obj A)
    ⊢ Eq (F.map (CategoryTheory.PreGaloisCategory.autMap f σ).hom (F.map f a)) (F. …
  -/
  simpa [-comp_autMap] using congrFun (F.congr_map (comp_autMap f σ)) a
  /-
    🎉 no goals
  -/


/-- `autMap` is uniquely characterized by making the canonical diagram commute. -/
lemma autMap_unique {A B : C} [IsConnected A] [IsGalois B] (f : A ⟶ B) (σ : Aut A)
    (τ : Aut B) (h : f ≫ τ.hom = σ.hom ≫ f) :
    autMap f σ = τ :=
  ((exists_autMap f σ).choose_spec.right τ h).symm


@[simp]
lemma autMap_id {A : C} [IsGalois A] : autMap (𝟙 A) = id :=
                                             /-
                                               C : Type u₁
                                               inst✝² : CategoryTheory.Category.{u₂, u₁} C
                                               inst✝¹ : CategoryTheory.GaloisCategory C
                                               A : C
                                               inst✝ : CategoryTheory.PreGaloisCategory.IsGalois A
                                               σ : CategoryTheory.Aut A
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id A)  …
                                             -/
  funext fun σ ↦ autMap_unique (𝟙 A) σ _ (by simp)
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
lemma autMap_comp {X Y Z : C} [IsConnected X] [IsGalois Y] [IsGalois Z] (f : X ⟶ Y)
    (g : Y ⟶ Z) : autMap (f ≫ g) = autMap g ∘ autMap f := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{u₂, u₁} C
    inst✝³ : CategoryTheory.GaloisCategory C
    X Y Z : C
    inst✝² : CategoryTheory.PreGaloisCategory.IsConnected X
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsGalois Y
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois Z
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.PreGaloisCategory.autMap (CategoryTheory.CategoryStruct.c …
  -/
  refine funext fun σ ↦ autMap_unique _ σ _ ?_
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{u₂, u₁} C
    inst✝³ : CategoryTheory.GaloisCategory C
    X Y Z : C
    inst✝² : CategoryTheory.PreGaloisCategory.IsConnected X
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsGalois Y
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois Z
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    σ : CategoryTheory.Aut X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
  -/
  rw [Function.comp_apply, Category.assoc, comp_autMap, ← Category.assoc]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{u₂, u₁} C
    inst✝³ : CategoryTheory.GaloisCategory C
    X Y Z : C
    inst✝² : CategoryTheory.PreGaloisCategory.IsConnected X
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsGalois Y
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois Z
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    σ : CategoryTheory.Aut X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- `autMap` is surjective, if the source is also Galois. -/
lemma autMap_surjective_of_isGalois {A B : C} [IsGalois A] [IsGalois B] (f : A ⟶ B) :
    Function.Surjective (autMap f) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    A B : C
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsGalois A
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
    f : Quiver.Hom A B
    ⊢ Function.Surjective (CategoryTheory.PreGaloisCategory.autMap f)
  -/
  intro σ
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    A B : C
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsGalois A
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
    f : Quiver.Hom A B
    σ : CategoryTheory.Aut B
    ⊢ Exists fun a => Eq (CategoryTheory.PreGaloisCategory.autMap f a) σ
  -/
  let F := GaloisCategory.getFiberFunctor C
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    A B : C
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsGalois A
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
    f : Quiver.Hom A B
    σ : CategoryTheory.Aut B
    F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
    ⊢ Exists fun a => Eq (CategoryTheory.PreGaloisCategory.autMap f a) σ
  -/
  obtain ⟨a⟩ := nonempty_fiber_of_isConnected F A
  /-
    case intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    A B : C
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsGalois A
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
    f : Quiver.Hom A B
    σ : CategoryTheory.Aut B
    F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
    a : ↑(F.obj A)
    ⊢ Exists fun a => Eq (CategoryTheory.PreGaloisCategory.autMap f a) σ
  -/
  obtain ⟨a', ha'⟩ := surjective_of_nonempty_fiber_of_isConnected F f (F.map σ.hom (F.map f a))
  /-
    case intro.intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    A B : C
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsGalois A
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
    f : Quiver.Hom A B
    σ : CategoryTheory.Aut B
    F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
    a a' : ↑(F.obj A)
    ha' : Eq (F.map f a') (F.map σ.hom (F.map f a))
    ⊢ Exists fun a => Eq (CategoryTheory.PreGaloisCategory.autMap f a) σ
  -/
  obtain ⟨τ, (hτ : F.map τ.hom a = a')⟩ := MulAction.exists_smul_eq (Aut A) a a'
  /-
    case intro.intro.intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    A B : C
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsGalois A
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
    f : Quiver.Hom A B
    σ : CategoryTheory.Aut B
    F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
    a a' : ↑(F.obj A)
    ha' : Eq (F.map f a') (F.map σ.hom (F.map f a))
    τ : CategoryTheory.Aut A
    hτ : Eq (F.map τ.hom a) a'
    ⊢ Exists fun a => Eq (CategoryTheory.PreGaloisCategory.autMap f a) σ
  -/
  use τ
  /-
    case h
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    A B : C
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsGalois A
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
    f : Quiver.Hom A B
    σ : CategoryTheory.Aut B
    F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
    a a' : ↑(F.obj A)
    ha' : Eq (F.map f a') (F.map σ.hom (F.map f a))
    τ : CategoryTheory.Aut A
    hτ : Eq (F.map τ.hom a) a'
    ⊢ Eq (CategoryTheory.PreGaloisCategory.autMap f τ) σ
  -/
  apply evaluation_aut_injective_of_isConnected F B (F.map f a)
  /-
    case h.a
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    A B : C
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsGalois A
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
    f : Quiver.Hom A B
    σ : CategoryTheory.Aut B
    F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
    a a' : ↑(F.obj A)
    ha' : Eq (F.map f a') (F.map σ.hom (F.map f a))
    τ : CategoryTheory.Aut A
    hτ : Eq (F.map τ.hom a) a'
    ⊢ Eq ((fun f_1 => F.map f_1.hom (F.map f a)) (CategoryTheory.PreGaloisCategory …
  -/
  simp [hτ, ha']
  /-
    🎉 no goals
  -/


@[simp]
lemma autMap_apply_mul {A B : C} [IsConnected A] [IsGalois B] (f : A ⟶ B) (σ τ : Aut A) :
    autMap f (σ * τ) = autMap f σ * autMap f τ := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    A B : C
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
    f : Quiver.Hom A B
    σ τ : CategoryTheory.Aut A
    ⊢ Eq (CategoryTheory.PreGaloisCategory.autMap f (HMul.hMul σ τ)) (HMul.hMul (C …
  -/
  let F := GaloisCategory.getFiberFunctor C
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    A B : C
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
    f : Quiver.Hom A B
    σ τ : CategoryTheory.Aut A
    F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
    ⊢ Eq (CategoryTheory.PreGaloisCategory.autMap f (HMul.hMul σ τ)) (HMul.hMul (C …
  -/
  obtain ⟨a⟩ := nonempty_fiber_of_isConnected F A
  /-
    case intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    A B : C
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
    f : Quiver.Hom A B
    σ τ : CategoryTheory.Aut A
    F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
    a : ↑(F.obj A)
    ⊢ Eq (CategoryTheory.PreGaloisCategory.autMap f (HMul.hMul σ τ)) (HMul.hMul (C …
  -/
  apply evaluation_aut_injective_of_isConnected F (B : C) (F.map f a)
  /-
    case intro.a
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    A B : C
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois B
    f : Quiver.Hom A B
    σ τ : CategoryTheory.Aut A
    F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
    a : ↑(F.obj A)
    ⊢ Eq ((fun f_1 => F.map f_1.hom (F.map f a)) (CategoryTheory.PreGaloisCategory …
  -/
  simp [Aut.Aut_mul_def]
  /-
    🎉 no goals
  -/


/-- `MonoidHom` version of `autMap`. -/
@[simps!]
noncomputable def autMapHom {A B : C} [IsConnected A] [IsGalois B] (f : A ⟶ B) :
     Aut A →* Aut B :=
  MonoidHom.mk' (autMap f) (autMap_apply_mul f)


