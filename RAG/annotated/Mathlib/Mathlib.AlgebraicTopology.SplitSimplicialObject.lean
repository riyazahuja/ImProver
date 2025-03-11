/-- The index set which appears in the definition of split simplicial objects. -/
def IndexSet (Δ : SimplexCategoryᵒᵖ) :=
  ΣΔ' : SimplexCategoryᵒᵖ, { α : Δ.unop ⟶ Δ'.unop // Epi α }


/-- The element in `Splitting.IndexSet Δ` attached to an epimorphism `f : Δ ⟶ Δ'`. -/
@[simps]
def mk {Δ Δ' : SimplexCategory} (f : Δ ⟶ Δ') [Epi f] : IndexSet (op Δ) :=
  ⟨op Δ', f, inferInstance⟩


/-- The epimorphism in `SimplexCategory` associated to `A : Splitting.IndexSet Δ` -/
def e :=
  A.2.1


instance : Epi A.e :=
  A.2.2


theorem ext' : A = ⟨A.1, ⟨A.e, A.2.2⟩⟩ := rfl


                                                                             /-
                                                                               C : Type u_1
                                                                               inst✝ : CategoryTheory.Category.{?u.827, u_1} C
                                                                               Δ : Opposite SimplexCategory
                                                                               A A₁ A₂ : SimplicialObject.Splitting.IndexSet Δ
                                                                               h₁ : Eq A₁.fst A₂.fst
                                                                               ⊢ Eq (Opposite.unop A₁.fst) (Opposite.unop A₂.fst)
                                                                             -/
theorem ext (A₁ A₂ : IndexSet Δ) (h₁ : A₁.1 = A₂.1) (h₂ : A₁.e ≫ eqToHom (by rw [h₁]) = A₂.e) :
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
    A₁ = A₂ := by
  /-
    Δ : Opposite SimplexCategory
    A₁ A₂ : SimplicialObject.Splitting.IndexSet Δ
    h₁ : Eq A₁.fst A₂.fst
    h₂ : Eq (CategoryTheory.CategoryStruct.comp A₁.e (CategoryTheory.eqToHom ⋯)) A …
    ⊢ Eq A₁ A₂
  -/
  rcases A₁ with ⟨Δ₁, ⟨α₁, hα₁⟩⟩
  /-
    case mk.mk
    Δ : Opposite SimplexCategory
    A₂ : SimplicialObject.Splitting.IndexSet Δ
    Δ₁ : Opposite SimplexCategory
    α₁ : Quiver.Hom (Opposite.unop Δ) (Opposite.unop Δ₁)
    hα₁ : CategoryTheory.Epi α₁
    h₁ : Eq ⟨Δ₁, ⟨α₁, hα₁⟩⟩.fst A₂.fst
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (SimplicialObject.Splitting.IndexS …
    ⊢ Eq ⟨Δ₁, ⟨α₁, hα₁⟩⟩ A₂
  -/
  rcases A₂ with ⟨Δ₂, ⟨α₂, hα₂⟩⟩
  /-
    case mk.mk.mk.mk
    Δ Δ₁ : Opposite SimplexCategory
    α₁ : Quiver.Hom (Opposite.unop Δ) (Opposite.unop Δ₁)
    hα₁ : CategoryTheory.Epi α₁
    Δ₂ : Opposite SimplexCategory
    α₂ : Quiver.Hom (Opposite.unop Δ) (Opposite.unop Δ₂)
    hα₂ : CategoryTheory.Epi α₂
    h₁ : Eq ⟨Δ₁, ⟨α₁, hα₁⟩⟩.fst ⟨Δ₂, ⟨α₂, hα₂⟩⟩.fst
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (SimplicialObject.Splitting.IndexS …
    ⊢ Eq ⟨Δ₁, ⟨α₁, hα₁⟩⟩ ⟨Δ₂, ⟨α₂, hα₂⟩⟩
  -/
  simp only at h₁
  /-
    case mk.mk.mk.mk
    Δ Δ₁ : Opposite SimplexCategory
    α₁ : Quiver.Hom (Opposite.unop Δ) (Opposite.unop Δ₁)
    hα₁ : CategoryTheory.Epi α₁
    Δ₂ : Opposite SimplexCategory
    α₂ : Quiver.Hom (Opposite.unop Δ) (Opposite.unop Δ₂)
    hα₂ : CategoryTheory.Epi α₂
    h₁ : Eq Δ₁ Δ₂
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (SimplicialObject.Splitting.IndexS …
    ⊢ Eq ⟨Δ₁, ⟨α₁, hα₁⟩⟩ ⟨Δ₂, ⟨α₂, hα₂⟩⟩
  -/
  subst h₁
  /-
    case mk.mk.mk.mk
    Δ Δ₁ : Opposite SimplexCategory
    α₁ : Quiver.Hom (Opposite.unop Δ) (Opposite.unop Δ₁)
    hα₁ : CategoryTheory.Epi α₁
    α₂ : Quiver.Hom (Opposite.unop Δ) (Opposite.unop Δ₁)
    hα₂ : CategoryTheory.Epi α₂
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (SimplicialObject.Splitting.IndexS …
    ⊢ Eq ⟨Δ₁, ⟨α₁, hα₁⟩⟩ ⟨Δ₁, ⟨α₂, hα₂⟩⟩
  -/
  simp only [eqToHom_refl, comp_id, IndexSet.e] at h₂
  /-
    case mk.mk.mk.mk
    Δ Δ₁ : Opposite SimplexCategory
    α₁ : Quiver.Hom (Opposite.unop Δ) (Opposite.unop Δ₁)
    hα₁ : CategoryTheory.Epi α₁
    α₂ : Quiver.Hom (Opposite.unop Δ) (Opposite.unop Δ₁)
    hα₂ : CategoryTheory.Epi α₂
    h₂ : Eq α₁ α₂
    ⊢ Eq ⟨Δ₁, ⟨α₁, hα₁⟩⟩ ⟨Δ₁, ⟨α₂, hα₂⟩⟩
  -/
  simp only [h₂]
  /-
    🎉 no goals
  -/


instance : Fintype (IndexSet Δ) :=
  Fintype.ofInjective
    (fun A =>
      ⟨⟨A.1.unop.len, Nat.lt_succ_iff.mpr (len_le_of_epi (inferInstance : Epi A.e))⟩,
        A.e.toOrderHom⟩ :
      IndexSet Δ → Sigma fun k : Fin (Δ.unop.len + 1) => Fin (Δ.unop.len + 1) → Fin (k + 1))
    (by
      /-
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.1568, u_1} C
        Δ : Opposite SimplexCategory
        A : SimplicialObject.Splitting.IndexSet Δ
        ⊢ Function.Injective fun A => ⟨⟨(Opposite.unop A.fst).len, ⋯⟩, ⇑(SimplexCatego …
      -/
      rintro ⟨Δ₁, α₁⟩ ⟨Δ₂, α₂⟩ h₁
      /-
        case mk.mk
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.1568, u_1} C
        Δ : Opposite SimplexCategory
        A : SimplicialObject.Splitting.IndexSet Δ
        Δ₁ : Opposite SimplexCategory
        α₁ : Subtype fun α => CategoryTheory.Epi α
        Δ₂ : Opposite SimplexCategory
        α₂ : Subtype fun α => CategoryTheory.Epi α
        h₁ : Eq ((fun A => ⟨⟨(Opposite.unop A.fst).len, ⋯⟩, ⇑(SimplexCategory.Hom.toOr …
        ⊢ Eq ⟨Δ₁, α₁⟩ ⟨Δ₂, α₂⟩
      -/
      induction' Δ₁ using Opposite.rec with Δ₁
      /-
        case mk.mk.op
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.1568, u_1} C
        Δ : Opposite SimplexCategory
        A : SimplicialObject.Splitting.IndexSet Δ
        Δ₂ : Opposite SimplexCategory
        α₂ : Subtype fun α => CategoryTheory.Epi α
        Δ₁ : SimplexCategory
        α₁ : Subtype fun α => CategoryTheory.Epi α
        h₁ : Eq ((fun A => ⟨⟨(Opposite.unop A.fst).len, ⋯⟩, ⇑(SimplexCategory.Hom.toOr …
        ⊢ Eq ⟨{ unop := Δ₁ }, α₁⟩ ⟨Δ₂, α₂⟩
      -/
      induction' Δ₂ using Opposite.rec with Δ₂
      /-
        case mk.mk.op.op
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.1568, u_1} C
        Δ : Opposite SimplexCategory
        A : SimplicialObject.Splitting.IndexSet Δ
        Δ₁ : SimplexCategory
        α₁ : Subtype fun α => CategoryTheory.Epi α
        Δ₂ : SimplexCategory
        α₂ : Subtype fun α => CategoryTheory.Epi α
        h₁ : Eq ((fun A => ⟨⟨(Opposite.unop A.fst).len, ⋯⟩, ⇑(SimplexCategory.Hom.toOr …
        ⊢ Eq ⟨{ unop := Δ₁ }, α₁⟩ ⟨{ unop := Δ₂ }, α₂⟩
      -/
      simp only [unop_op, Sigma.mk.inj_iff, Fin.mk.injEq] at h₁
      have h₂ : Δ₁ = Δ₂ := by
        ext1
        simpa only [Fin.mk_eq_mk] using h₁.1
      /-
        case mk.mk.op.op
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.1568, u_1} C
        Δ : Opposite SimplexCategory
        A : SimplicialObject.Splitting.IndexSet Δ
        Δ₁ : SimplexCategory
        α₁ : Subtype fun α => CategoryTheory.Epi α
        Δ₂ : SimplexCategory
        α₂ : Subtype fun α => CategoryTheory.Epi α
        h₁ : And (Eq Δ₁.len Δ₂.len) (HEq ⇑(SimplexCategory.Hom.toOrderHom (SimplicialO …
        h₂ : Eq Δ₁ Δ₂
        ⊢ Eq ⟨{ unop := Δ₁ }, α₁⟩ ⟨{ unop := Δ₂ }, α₂⟩
      -/
      subst h₂
      /-
        case mk.mk.op.op
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.1568, u_1} C
        Δ : Opposite SimplexCategory
        A : SimplicialObject.Splitting.IndexSet Δ
        Δ₁ : SimplexCategory
        α₁ α₂ : Subtype fun α => CategoryTheory.Epi α
        h₁ : And (Eq Δ₁.len Δ₁.len) (HEq ⇑(SimplexCategory.Hom.toOrderHom (SimplicialO …
        ⊢ Eq ⟨{ unop := Δ₁ }, α₁⟩ ⟨{ unop := Δ₁ }, α₂⟩
      -/
      refine ext _ _ rfl ?_
      /-
        case mk.mk.op.op
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.1568, u_1} C
        Δ : Opposite SimplexCategory
        A : SimplicialObject.Splitting.IndexSet Δ
        Δ₁ : SimplexCategory
        α₁ α₂ : Subtype fun α => CategoryTheory.Epi α
        h₁ : And (Eq Δ₁.len Δ₁.len) (HEq ⇑(SimplexCategory.Hom.toOrderHom (SimplicialO …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplicialObject.Splitting.IndexSet. …
      -/
      ext : 2
      /-
        case mk.mk.op.op.a.h
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.1568, u_1} C
        Δ : Opposite SimplexCategory
        A : SimplicialObject.Splitting.IndexSet Δ
        Δ₁ : SimplexCategory
        α₁ α₂ : Subtype fun α => CategoryTheory.Epi α
        h₁ : And (Eq Δ₁.len Δ₁.len) (HEq ⇑(SimplexCategory.Hom.toOrderHom (SimplicialO …
        ⊢ Eq ⇑(SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Sim …
      -/
      exact eq_of_heq h₁.2)
      /-
        🎉 no goals
      -/


/-- The distinguished element in `Splitting.IndexSet Δ` which corresponds to the
identity of `Δ`. -/
@[simps]
def id : IndexSet Δ :=
               /-
                 C : Type u_1
                 inst✝ : CategoryTheory.Category.{?u.3310, u_1} C
                 Δ : Opposite SimplexCategory
                 A : SimplicialObject.Splitting.IndexSet Δ
                 ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.id (Opposite.unop Δ))
               -/
  ⟨Δ, ⟨𝟙 _, by infer_instance⟩⟩
               /-
                 🎉 no goals
               -/


instance : Inhabited (IndexSet Δ) :=
  ⟨id Δ⟩


/-- The condition that an element `Splitting.IndexSet Δ` is the distinguished
element `Splitting.IndexSet.Id Δ`. -/
@[simp]
def EqId : Prop :=
  A = id _


theorem eqId_iff_eq : A.EqId ↔ A.1 = Δ := by
  /-
    Δ : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    ⊢ Iff A.EqId (Eq A.fst Δ)
  -/
  constructor
    /-
      case mp
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      ⊢ A.EqId → Eq A.fst Δ
    -/
  · intro h
    /-
      case mp
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      h : A.EqId
      ⊢ Eq A.fst Δ
    -/
    dsimp at h
    /-
      case mp
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      h : Eq A (SimplicialObject.Splitting.IndexSet.id Δ)
      ⊢ Eq A.fst Δ
    -/
    rw [h]
    /-
      case mp
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      h : Eq A (SimplicialObject.Splitting.IndexSet.id Δ)
      ⊢ Eq (SimplicialObject.Splitting.IndexSet.id Δ).fst Δ
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case mpr
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      ⊢ Eq A.fst Δ → A.EqId
    -/
  · intro h
    /-
      case mpr
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      h : Eq A.fst Δ
      ⊢ A.EqId
    -/
    rcases A with ⟨_, ⟨f, hf⟩⟩
    /-
      case mpr.mk.mk
      Δ fst✝ : Opposite SimplexCategory
      f : Quiver.Hom (Opposite.unop Δ) (Opposite.unop fst✝)
      hf : CategoryTheory.Epi f
      h : Eq ⟨fst✝, ⟨f, hf⟩⟩.fst Δ
      ⊢ SimplicialObject.Splitting.IndexSet.EqId ⟨fst✝, ⟨f, hf⟩⟩
    -/
    simp only at h
    /-
      case mpr.mk.mk
      Δ fst✝ : Opposite SimplexCategory
      f : Quiver.Hom (Opposite.unop Δ) (Opposite.unop fst✝)
      hf : CategoryTheory.Epi f
      h : Eq fst✝ Δ
      ⊢ SimplicialObject.Splitting.IndexSet.EqId ⟨fst✝, ⟨f, hf⟩⟩
    -/
    subst h
    /-
      case mpr.mk.mk
      fst✝ : Opposite SimplexCategory
      f : Quiver.Hom (Opposite.unop fst✝) (Opposite.unop fst✝)
      hf : CategoryTheory.Epi f
      ⊢ SimplicialObject.Splitting.IndexSet.EqId ⟨fst✝, ⟨f, hf⟩⟩
    -/
    refine ext _ _ rfl ?_
    /-
      case mpr.mk.mk
      fst✝ : Opposite SimplexCategory
      f : Quiver.Hom (Opposite.unop fst✝) (Opposite.unop fst✝)
      hf : CategoryTheory.Epi f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplicialObject.Splitting.IndexSet. …
    -/
    haveI := hf
    /-
      case mpr.mk.mk
      fst✝ : Opposite SimplexCategory
      f : Quiver.Hom (Opposite.unop fst✝) (Opposite.unop fst✝)
      hf this : CategoryTheory.Epi f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplicialObject.Splitting.IndexSet. …
    -/
    simp only [eqToHom_refl, comp_id]
    /-
      case mpr.mk.mk
      fst✝ : Opposite SimplexCategory
      f : Quiver.Hom (Opposite.unop fst✝) (Opposite.unop fst✝)
      hf this : CategoryTheory.Epi f
      ⊢ Eq (SimplicialObject.Splitting.IndexSet.e ⟨fst✝, ⟨f, hf⟩⟩) (SimplicialObject …
    -/
    exact eq_id_of_epi f
    /-
      🎉 no goals
    -/


theorem eqId_iff_len_eq : A.EqId ↔ A.1.unop.len = Δ.unop.len := by
  /-
    Δ : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    ⊢ Iff A.EqId (Eq (Opposite.unop A.fst).len (Opposite.unop Δ).len)
  -/
  rw [eqId_iff_eq]
  /-
    Δ : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    ⊢ Iff (Eq A.fst Δ) (Eq (Opposite.unop A.fst).len (Opposite.unop Δ).len)
  -/
  constructor
    /-
      case mp
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      ⊢ Eq A.fst Δ → Eq (Opposite.unop A.fst).len (Opposite.unop Δ).len
    -/
  · intro h
    /-
      case mp
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      h : Eq A.fst Δ
      ⊢ Eq (Opposite.unop A.fst).len (Opposite.unop Δ).len
    -/
    rw [h]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      ⊢ Eq (Opposite.unop A.fst).len (Opposite.unop Δ).len → Eq A.fst Δ
    -/
  · intro h
    /-
      case mpr
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      h : Eq (Opposite.unop A.fst).len (Opposite.unop Δ).len
      ⊢ Eq A.fst Δ
    -/
    rw [← unop_inj_iff]
    /-
      case mpr
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      h : Eq (Opposite.unop A.fst).len (Opposite.unop Δ).len
      ⊢ Eq (Opposite.unop A.fst) (Opposite.unop Δ)
    -/
    ext
    /-
      case mpr.a
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      h : Eq (Opposite.unop A.fst).len (Opposite.unop Δ).len
      ⊢ Eq (Opposite.unop A.fst).len (Opposite.unop Δ).len
    -/
    exact h
    /-
      🎉 no goals
    -/


theorem eqId_iff_len_le : A.EqId ↔ Δ.unop.len ≤ A.1.unop.len := by
  /-
    Δ : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    ⊢ Iff A.EqId (LE.le (Opposite.unop Δ).len (Opposite.unop A.fst).len)
  -/
  rw [eqId_iff_len_eq]
  /-
    Δ : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    ⊢ Iff (Eq (Opposite.unop A.fst).len (Opposite.unop Δ).len) (LE.le (Opposite.un …
  -/
  constructor
    /-
      case mp
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      ⊢ Eq (Opposite.unop A.fst).len (Opposite.unop Δ).len → LE.le (Opposite.unop Δ) …
    -/
  · intro h
    /-
      case mp
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      h : Eq (Opposite.unop A.fst).len (Opposite.unop Δ).len
      ⊢ LE.le (Opposite.unop Δ).len (Opposite.unop A.fst).len
    -/
    rw [h]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      ⊢ LE.le (Opposite.unop Δ).len (Opposite.unop A.fst).len → Eq (Opposite.unop A. …
    -/
  · exact le_antisymm (len_le_of_epi (inferInstance : Epi A.e))
    /-
      🎉 no goals
    -/


theorem eqId_iff_mono : A.EqId ↔ Mono A.e := by
  /-
    Δ : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    ⊢ Iff A.EqId (CategoryTheory.Mono A.e)
  -/
  constructor
    /-
      case mp
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      ⊢ A.EqId → CategoryTheory.Mono A.e
    -/
  · intro h
    /-
      case mp
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      h : A.EqId
      ⊢ CategoryTheory.Mono A.e
    -/
    dsimp at h
    /-
      case mp
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      h : Eq A (SimplicialObject.Splitting.IndexSet.id Δ)
      ⊢ CategoryTheory.Mono A.e
    -/
    subst h
    /-
      case mp
      Δ : Opposite SimplexCategory
      ⊢ CategoryTheory.Mono (SimplicialObject.Splitting.IndexSet.id Δ).e
    -/
    dsimp only [id, e]
    /-
      case mp
      Δ : Opposite SimplexCategory
      ⊢ CategoryTheory.Mono (CategoryTheory.CategoryStruct.id (Opposite.unop Δ))
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      ⊢ CategoryTheory.Mono A.e → A.EqId
    -/
  · intro h
    /-
      case mpr
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      h : CategoryTheory.Mono A.e
      ⊢ A.EqId
    -/
    rw [eqId_iff_len_le]
    /-
      case mpr
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      h : CategoryTheory.Mono A.e
      ⊢ LE.le (Opposite.unop Δ).len (Opposite.unop A.fst).len
    -/
    exact len_le_of_mono h
    /-
      🎉 no goals
    -/


/-- Given `A : IndexSet Δ₁`, if `p.unop : unop Δ₂ ⟶ unop Δ₁` is an epi, this
is the obvious element in `A : IndexSet Δ₂` associated to the composition
of epimorphisms `p.unop ≫ A.e`. -/
@[simps]
def epiComp {Δ₁ Δ₂ : SimplexCategoryᵒᵖ} (A : IndexSet Δ₁) (p : Δ₁ ⟶ Δ₂) [Epi p.unop] :
    IndexSet Δ₂ :=
  ⟨A.1, ⟨p.unop ≫ A.e, epi_comp _ _⟩⟩



/-- When `A : IndexSet Δ` and `θ : Δ → Δ'` is a morphism in `SimplexCategoryᵒᵖ`,
an element in `IndexSet Δ'` can be defined by using the epi-mono factorisation
of `θ.unop ≫ A.e`. -/
def pull : IndexSet Δ' :=
  mk (factorThruImage (θ.unop ≫ A.e))


@[reassoc]
theorem fac_pull : (A.pull θ).e ≫ image.ι (θ.unop ≫ A.e) = θ.unop ≫ A.e :=
  image.fac _


/-- Given a sequences of objects `N : ℕ → C` in a category `C`, this is
a family of objects indexed by the elements `A : Splitting.IndexSet Δ`.
The `Δ`-simplices of a split simplicial objects shall identify to the
coproduct of objects in such a family. -/
@[simp, nolint unusedArguments]
def summand (A : IndexSet Δ) : C :=
  N A.1.unop.len


/-- The cofan for `summand N Δ` induced by morphisms `N n ⟶ X_ [n]` for all `n : ℕ`. -/
def cofan' (Δ : SimplexCategoryᵒᵖ) : Cofan (summand N Δ) :=
  Cofan.mk (X.obj Δ) (fun A => φ A.1.unop.len ≫ X.map A.e.op)


/-- A splitting of a simplicial object `X` consists of the datum of a sequence
of objects `N`, a sequence of morphisms `ι : N n ⟶ X _[n]` such that
for all `Δ : SimplexCategoryᵒᵖ`, the canonical map `Splitting.map X ι Δ`
is an isomorphism. -/
structure Splitting (X : SimplicialObject C) where
  /-- The "nondegenerate simplices" `N n` for all `n : ℕ`. -/
  N : ℕ → C
  /-- The "inclusion" `N n ⟶ X _[n]` for all `n : ℕ`. -/
  ι : ∀ n, N n ⟶ X _[n]
  /-- For each `Δ`, `X.obj Δ` identifies to the coproduct of the objects `N A.1.unop.len`
  for all `A : IndexSet Δ`. -/
  isColimit' : ∀ Δ : SimplexCategoryᵒᵖ, IsColimit (Splitting.cofan' N X ι Δ)


/-- The cofan for `summand s.N Δ` induced by a splitting of a simplicial object. -/
def cofan (Δ : SimplexCategoryᵒᵖ) : Cofan (summand s.N Δ) :=
  Cofan.mk (X.obj Δ) (fun A => s.ι A.1.unop.len ≫ X.map A.e.op)


/-- The cofan `s.cofan Δ` is colimit. -/
def isColimit (Δ : SimplexCategoryᵒᵖ) : IsColimit (s.cofan Δ) := s.isColimit' Δ


@[reassoc]
theorem cofan_inj_eq {Δ : SimplexCategoryᵒᵖ} (A : IndexSet Δ) :
    (s.cofan Δ).inj  A = s.ι A.1.unop.len ≫ X.map A.e.op := rfl


theorem cofan_inj_id (n : ℕ) : (s.cofan _).inj (IndexSet.id (op [n])) = s.ι n := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    n : Nat
    ⊢ Eq ((s.cofan { unop := SimplexCategory.mk n }).inj (SimplicialObject.Splitti …
  -/
  erw [cofan_inj_eq, X.map_id, comp_id]
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    n : Nat
    ⊢ Eq (s.ι (Opposite.unop (SimplicialObject.Splitting.IndexSet.id { unop := Sim …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- As it is stated in `Splitting.hom_ext`, a morphism `f : X ⟶ Y` from a split
simplicial object to any simplicial object is determined by its restrictions
`s.φ f n : s.N n ⟶ Y _[n]` to the distinguished summands in each degree `n`. -/
@[simp]
def φ (f : X ⟶ Y) (n : ℕ) : s.N n ⟶ Y _[n] :=
  s.ι n ≫ f.app (op [n])


@[reassoc (attr := simp)]
theorem cofan_inj_comp_app (f : X ⟶ Y) {Δ : SimplexCategoryᵒᵖ} (A : IndexSet Δ) :
    (s.cofan Δ).inj A ≫ f.app Δ = s.φ f A.1.unop.len ≫ Y.map A.e.op := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    f : Quiver.Hom X Y
    Δ : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan Δ).inj A) (f.app Δ)) (Categ …
  -/
  simp only [cofan_inj_eq_assoc, φ, assoc]
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    f : Quiver.Hom X Y
    Δ : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι (Opposite.unop A.fst).len) (Cate …
  -/
  rw [NatTrans.naturality]
  /-
    🎉 no goals
  -/


theorem hom_ext' {Z : C} {Δ : SimplexCategoryᵒᵖ} (f g : X.obj Δ ⟶ Z)
    (h : ∀ A : IndexSet Δ, (s.cofan Δ).inj A ≫ f = (s.cofan Δ).inj A ≫ g) : f = g :=
  Cofan.IsColimit.hom_ext (s.isColimit Δ) _ _ h


theorem hom_ext (f g : X ⟶ Y) (h : ∀ n : ℕ, s.φ f n = s.φ g n) : f = g := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    f g : Quiver.Hom X Y
    h : ∀ (n : Nat), Eq (s.φ f n) (s.φ g n)
    ⊢ Eq f g
  -/
  ext Δ
  /-
    case h
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    f g : Quiver.Hom X Y
    h : ∀ (n : Nat), Eq (s.φ f n) (s.φ g n)
    Δ : Opposite SimplexCategory
    ⊢ Eq (f.app Δ) (g.app Δ)
  -/
  apply s.hom_ext'
  /-
    case h.h
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    f g : Quiver.Hom X Y
    h : ∀ (n : Nat), Eq (s.φ f n) (s.φ g n)
    Δ : Opposite SimplexCategory
    ⊢ ∀ (A : SimplicialObject.Splitting.IndexSet Δ), Eq (CategoryTheory.CategorySt …
  -/
  intro A
  /-
    case h.h
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    f g : Quiver.Hom X Y
    h : ∀ (n : Nat), Eq (s.φ f n) (s.φ g n)
    Δ : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan Δ).inj A) (f.app Δ)) (Categ …
  -/
  induction' Δ using Opposite.rec with Δ
  /-
    case h.h.op
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    f g : Quiver.Hom X Y
    h : ∀ (n : Nat), Eq (s.φ f n) (s.φ g n)
    Δ : SimplexCategory
    A : SimplicialObject.Splitting.IndexSet { unop := Δ }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan { unop := Δ }).inj A) (f.ap …
  -/
  induction' Δ using SimplexCategory.rec with n
  /-
    case h.h.op.h
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    f g : Quiver.Hom X Y
    h : ∀ (n : Nat), Eq (s.φ f n) (s.φ g n)
    n : Nat
    A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan { unop := SimplexCategory.m …
  -/
  dsimp
  /-
    case h.h.op.h
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    f g : Quiver.Hom X Y
    h : ∀ (n : Nat), Eq (s.φ f n) (s.φ g n)
    n : Nat
    A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan { unop := SimplexCategory.m …
  -/
  simp only [s.cofan_inj_comp_app, h]
  /-
    🎉 no goals
  -/


/-- The map `X.obj Δ ⟶ Z` obtained by providing a family of morphisms on all the
terms of decomposition given by a splitting `s : Splitting X`  -/
def desc {Z : C} (Δ : SimplexCategoryᵒᵖ) (F : ∀ A : IndexSet Δ, s.N A.1.unop.len ⟶ Z) :
    X.obj Δ ⟶ Z :=
  Cofan.IsColimit.desc (s.isColimit Δ) F


@[reassoc (attr := simp)]
theorem ι_desc {Z : C} (Δ : SimplexCategoryᵒᵖ) (F : ∀ A : IndexSet Δ, s.N A.1.unop.len ⟶ Z)
    (A : IndexSet Δ) : (s.cofan Δ).inj A ≫ s.desc Δ F = F A := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    Z : C
    Δ : Opposite SimplexCategory
    F : (A : SimplicialObject.Splitting.IndexSet Δ) → Quiver.Hom (s.N (Opposite.un …
    A : SimplicialObject.Splitting.IndexSet Δ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan Δ).inj A) (s.desc Δ F)) (F A)
  -/
  apply Cofan.IsColimit.fac
  /-
    🎉 no goals
  -/


/-- A simplicial object that is isomorphic to a split simplicial object is split. -/
@[simps]
def ofIso (e : X ≅ Y) : Splitting Y where
  N := s.N
  ι n := s.ι n ≫ e.hom.app (op [n])
  isColimit' Δ := IsColimit.ofIsoColimit (s.isColimit Δ ) (Cofan.ext (e.app Δ)
                 /-
                   C : Type u_1
                   inst✝ : CategoryTheory.Category.{?u.26656, u_1} C
                   X Y : CategoryTheory.SimplicialObject C
                   s : SimplicialObject.Splitting X
                   e : CategoryTheory.Iso X Y
                   Δ : Opposite SimplexCategory
                   A : SimplicialObject.Splitting.IndexSet Δ
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan Δ).inj A) (e.app Δ).hom) (( …
                 -/
    (fun A => by simp [cofan, cofan']))
                 /-
                   🎉 no goals
                 -/


@[reassoc]
theorem cofan_inj_epi_naturality {Δ₁ Δ₂ : SimplexCategoryᵒᵖ} (A : IndexSet Δ₁) (p : Δ₁ ⟶ Δ₂)
    [Epi p.unop] : (s.cofan Δ₁).inj A ≫ X.map p = (s.cofan Δ₂).inj (A.epiComp p) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    Δ₁ Δ₂ : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ₁
    p : Quiver.Hom Δ₁ Δ₂
    inst✝ : CategoryTheory.Epi p.unop
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan Δ₁).inj A) (X.map p)) ((s.c …
  -/
  dsimp [cofan]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    Δ₁ Δ₂ : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ₁
    p : Quiver.Hom Δ₁ Δ₂
    inst✝ : CategoryTheory.Epi p.unop
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [assoc, ← X.map_comp]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    Δ₁ Δ₂ : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ₁
    p : Quiver.Hom Δ₁ Δ₂
    inst✝ : CategoryTheory.Epi p.unop
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι (Opposite.unop A.fst).len) (X.ma …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The category `SimplicialObject.Split C` is the category of simplicial objects
in `C` equipped with a splitting, and morphisms are morphisms of simplicial objects
which are compatible with the splittings. -/
@[ext]
structure Split where
  /-- the underlying simplicial object -/
  X : SimplicialObject C
  /-- a splitting of the simplicial object -/
  s : Splitting X


/-- The object in `SimplicialObject.Split C` attached to a splitting `s : Splitting X`
of a simplicial object `X`. -/
@[simps]
def mk' {X : SimplicialObject C} (s : Splitting X) : Split C :=
  ⟨X, s⟩

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]

/-- Morphisms in `SimplicialObject.Split C` are morphisms of simplicial objects that
are compatible with the splittings. -/
structure Hom (S₁ S₂ : Split C) where
  /-- the morphism between the underlying simplicial objects -/
  F : S₁.X ⟶ S₂.X
  /-- the morphism between the "nondegenerate" `n`-simplices for all `n : ℕ` -/
  f : ∀ n : ℕ, S₁.s.N n ⟶ S₂.s.N n
  comm : ∀ n : ℕ, S₁.s.ι n ≫ F.app (op [n]) = f n ≫ S₂.s.ι n := by aesop_cat


@[ext]
theorem Hom.ext {S₁ S₂ : Split C} (Φ₁ Φ₂ : Hom S₁ S₂) (h : ∀ n : ℕ, Φ₁.f n = Φ₂.f n) : Φ₁ = Φ₂ := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    S₁ S₂ : SimplicialObject.Split C
    Φ₁ Φ₂ : S₁.Hom S₂
    h : ∀ (n : Nat), Eq (Φ₁.f n) (Φ₂.f n)
    ⊢ Eq Φ₁ Φ₂
  -/
  rcases Φ₁ with ⟨F₁, f₁, c₁⟩
  /-
    case mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    S₁ S₂ : SimplicialObject.Split C
    Φ₂ : S₁.Hom S₂
    F₁ : Quiver.Hom S₁.X S₂.X
    f₁ : (n : Nat) → Quiver.Hom (S₁.s.N n) (S₂.s.N n)
    c₁ : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (S₁.s.ι n) (F₁.app {  …
    h : ∀ (n : Nat), Eq ({ F := F₁, f := f₁, comm := c₁ }.f n) (Φ₂.f n)
    ⊢ Eq { F := F₁, f := f₁, comm := c₁ } Φ₂
  -/
  rcases Φ₂ with ⟨F₂, f₂, c₂⟩
  have h' : f₁ = f₂ := by
    ext
    apply h
  /-
    case mk.mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    S₁ S₂ : SimplicialObject.Split C
    F₁ : Quiver.Hom S₁.X S₂.X
    f₁ : (n : Nat) → Quiver.Hom (S₁.s.N n) (S₂.s.N n)
    c₁ : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (S₁.s.ι n) (F₁.app {  …
    F₂ : Quiver.Hom S₁.X S₂.X
    f₂ : (n : Nat) → Quiver.Hom (S₁.s.N n) (S₂.s.N n)
    c₂ : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (S₁.s.ι n) (F₂.app {  …
    h : ∀ (n : Nat), Eq ({ F := F₁, f := f₁, comm := c₁ }.f n) ({ F := F₂, f := f₂ …
    h' : Eq f₁ f₂
    ⊢ Eq { F := F₁, f := f₁, comm := c₁ } { F := F₂, f := f₂, comm := c₂ }
  -/
  subst h'
  /-
    case mk.mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    S₁ S₂ : SimplicialObject.Split C
    F₁ : Quiver.Hom S₁.X S₂.X
    f₁ : (n : Nat) → Quiver.Hom (S₁.s.N n) (S₂.s.N n)
    c₁ : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (S₁.s.ι n) (F₁.app {  …
    F₂ : Quiver.Hom S₁.X S₂.X
    c₂ : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (S₁.s.ι n) (F₂.app {  …
    h : ∀ (n : Nat), Eq ({ F := F₁, f := f₁, comm := c₁ }.f n) ({ F := F₂, f := f₁ …
    ⊢ Eq { F := F₁, f := f₁, comm := c₁ } { F := F₂, f := f₁, comm := c₂ }
  -/
  simp only [mk.injEq, and_true]
  /-
    case mk.mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    S₁ S₂ : SimplicialObject.Split C
    F₁ : Quiver.Hom S₁.X S₂.X
    f₁ : (n : Nat) → Quiver.Hom (S₁.s.N n) (S₂.s.N n)
    c₁ : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (S₁.s.ι n) (F₁.app {  …
    F₂ : Quiver.Hom S₁.X S₂.X
    c₂ : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (S₁.s.ι n) (F₂.app {  …
    h : ∀ (n : Nat), Eq ({ F := F₁, f := f₁, comm := c₁ }.f n) ({ F := F₂, f := f₁ …
    ⊢ Eq F₁ F₂
  -/
  apply S₁.s.hom_ext
  /-
    case mk.mk.h
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    S₁ S₂ : SimplicialObject.Split C
    F₁ : Quiver.Hom S₁.X S₂.X
    f₁ : (n : Nat) → Quiver.Hom (S₁.s.N n) (S₂.s.N n)
    c₁ : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (S₁.s.ι n) (F₁.app {  …
    F₂ : Quiver.Hom S₁.X S₂.X
    c₂ : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (S₁.s.ι n) (F₂.app {  …
    h : ∀ (n : Nat), Eq ({ F := F₁, f := f₁, comm := c₁ }.f n) ({ F := F₂, f := f₁ …
    ⊢ ∀ (n : Nat), Eq (S₁.s.φ F₁ n) (S₁.s.φ F₂ n)
  -/
  intro n
  /-
    case mk.mk.h
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    S₁ S₂ : SimplicialObject.Split C
    F₁ : Quiver.Hom S₁.X S₂.X
    f₁ : (n : Nat) → Quiver.Hom (S₁.s.N n) (S₂.s.N n)
    c₁ : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (S₁.s.ι n) (F₁.app {  …
    F₂ : Quiver.Hom S₁.X S₂.X
    c₂ : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (S₁.s.ι n) (F₂.app {  …
    h : ∀ (n : Nat), Eq ({ F := F₁, f := f₁, comm := c₁ }.f n) ({ F := F₂, f := f₁ …
    n : Nat
    ⊢ Eq (S₁.s.φ F₁ n) (S₁.s.φ F₂ n)
  -/
  dsimp
  /-
    case mk.mk.h
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    S₁ S₂ : SimplicialObject.Split C
    F₁ : Quiver.Hom S₁.X S₂.X
    f₁ : (n : Nat) → Quiver.Hom (S₁.s.N n) (S₂.s.N n)
    c₁ : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (S₁.s.ι n) (F₁.app {  …
    F₂ : Quiver.Hom S₁.X S₂.X
    c₂ : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (S₁.s.ι n) (F₂.app {  …
    h : ∀ (n : Nat), Eq ({ F := F₁, f := f₁, comm := c₁ }.f n) ({ F := F₂, f := f₁ …
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S₁.s.ι n) (F₁.app { unop := SimplexC …
  -/
  rw [c₁, c₂]
  /-
    🎉 no goals
  -/


attribute [simp, reassoc] Hom.comm


instance : Category (Split C) where
  Hom := Split.Hom
  id S :=
    { F := 𝟙 _
      f := fun _ => 𝟙 _ }
  comp Φ₁₂ Φ₂₃ :=
    { F := Φ₁₂.F ≫ Φ₂₃.F
      f := fun n => Φ₁₂.f n ≫ Φ₂₃.f n
      comm := fun n => by
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.35831, u_1} C
          X✝ Y✝ Z✝ : SimplicialObject.Split C
          Φ₁₂ : Quiver.Hom X✝ Y✝
          Φ₂₃ : Quiver.Hom Y✝ Z✝
          n : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (X✝.s.ι n) ((CategoryTheory.CategoryS …
        -/
        dsimp
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.35831, u_1} C
          X✝ Y✝ Z✝ : SimplicialObject.Split C
          Φ₁₂ : Quiver.Hom X✝ Y✝
          Φ₂₃ : Quiver.Hom Y✝ Z✝
          n : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (X✝.s.ι n) (CategoryTheory.CategorySt …
        -/
        simp only [assoc, Split.Hom.comm_assoc, Split.Hom.comm] }
        /-
          🎉 no goals
        -/


@[ext]
theorem hom_ext {S₁ S₂ : Split C} (Φ₁ Φ₂ : S₁ ⟶ S₂) (h : ∀ n : ℕ, Φ₁.f n = Φ₂.f n) : Φ₁ = Φ₂ :=
  Hom.ext _ _ h


                                                                                      /-
                                                                                        C : Type u_1
                                                                                        inst✝ : CategoryTheory.Category.{u_2, u_1} C
                                                                                        S₁ S₂ : SimplicialObject.Split C
                                                                                        Φ₁ Φ₂ : Quiver.Hom S₁ S₂
                                                                                        h : Eq Φ₁ Φ₂
                                                                                        ⊢ Eq Φ₁.f Φ₂.f
                                                                                      -/
theorem congr_F {S₁ S₂ : Split C} {Φ₁ Φ₂ : S₁ ⟶ S₂} (h : Φ₁ = Φ₂) : Φ₁.f = Φ₂.f := by rw [h]
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


theorem congr_f {S₁ S₂ : Split C} {Φ₁ Φ₂ : S₁ ⟶ S₂} (h : Φ₁ = Φ₂) (n : ℕ) : Φ₁.f n = Φ₂.f n := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    S₁ S₂ : SimplicialObject.Split C
    Φ₁ Φ₂ : Quiver.Hom S₁ S₂
    h : Eq Φ₁ Φ₂
    n : Nat
    ⊢ Eq (Φ₁.f n) (Φ₂.f n)
  -/
  rw [h]
  /-
    🎉 no goals
  -/


@[simp]
theorem id_F (S : Split C) : (𝟙 S : S ⟶ S).F = 𝟙 S.X :=
  rfl


@[simp]
theorem id_f (S : Split C) (n : ℕ) : (𝟙 S : S ⟶ S).f n = 𝟙 (S.s.N n) :=
  rfl


@[simp]
theorem comp_F {S₁ S₂ S₃ : Split C} (Φ₁₂ : S₁ ⟶ S₂) (Φ₂₃ : S₂ ⟶ S₃) :
    (Φ₁₂ ≫ Φ₂₃).F = Φ₁₂.F ≫ Φ₂₃.F :=
  rfl


@[simp]
theorem comp_f {S₁ S₂ S₃ : Split C} (Φ₁₂ : S₁ ⟶ S₂) (Φ₂₃ : S₂ ⟶ S₃) (n : ℕ) :
    (Φ₁₂ ≫ Φ₂₃).f n = Φ₁₂.f n ≫ Φ₂₃.f n :=
  rfl


@[reassoc (attr := simp 1100)]
theorem cofan_inj_naturality_symm {S₁ S₂ : Split C} (Φ : S₁ ⟶ S₂) {Δ : SimplexCategoryᵒᵖ}
    (A : Splitting.IndexSet Δ) :
    (S₁.s.cofan Δ).inj A ≫ Φ.F.app Δ = Φ.f A.1.unop.len ≫ (S₂.s.cofan Δ).inj A := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    S₁ S₂ : SimplicialObject.Split C
    Φ : Quiver.Hom S₁ S₂
    Δ : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((S₁.s.cofan Δ).inj A) (Φ.F.app Δ)) ( …
  -/
  rw [S₁.s.cofan_inj_eq, S₂.s.cofan_inj_eq, assoc, Φ.F.naturality, ← Φ.comm_assoc]
  /-
    🎉 no goals
  -/


/-- The functor `SimplicialObject.Split C ⥤ SimplicialObject C` which forgets
the splitting. -/
@[simps]
def forget : Split C ⥤ SimplicialObject C where
  obj S := S.X
  map Φ := Φ.F


/-- The functor `SimplicialObject.Split C ⥤ C` which sends a simplicial object equipped
with a splitting to its nondegenerate `n`-simplices. -/
@[simps]
def evalN (n : ℕ) : Split C ⥤ C where
  obj S := S.s.N n
  map Φ := Φ.f n


/-- The inclusion of each summand in the coproduct decomposition of simplices
in split simplicial objects is a natural transformation of functors
`SimplicialObject.Split C ⥤ C` -/
@[simps]
def natTransCofanInj {Δ : SimplexCategoryᵒᵖ} (A : Splitting.IndexSet Δ) :
    evalN C A.1.unop.len ⟶ forget C ⋙ (evaluation SimplexCategoryᵒᵖ C).obj Δ where
  app S := (S.s.cofan Δ).inj A
  naturality _ _ Φ := (cofan_inj_naturality_symm Φ A).symm


