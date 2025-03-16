/-- The trivial case if `X` is connected. -/
private lemma has_decomp_connected_components_aux_conn (X : C) [IsConnected X] :
    ∃ (ι : Type) (f : ι → C) (g : (i : ι) → (f i) ⟶ X) (_ : IsColimit (Cofan.mk X g)),
    (∀ i, IsConnected (f i)) ∧ Finite ι := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    ⊢ Exists fun ι => Exists fun f => Exists fun g => Exists fun x => And (∀ (i :  …
  -/
  refine ⟨Unit, fun _ ↦ X, fun _ ↦ 𝟙 X, mkCofanColimit _ (fun s ↦ s.inj ()), ?_⟩
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    ⊢ And (∀ (i : Unit), CategoryTheory.PreGaloisCategory.IsConnected ((fun x => X …
  -/
  exact ⟨fun _ ↦ inferInstance, inferInstance⟩
  /-
    🎉 no goals
  -/


/-- The trivial case if `X` is initial. -/
private lemma has_decomp_connected_components_aux_initial (X : C) (h : IsInitial X) :
    ∃ (ι : Type) (f : ι → C) (g : (i : ι) → (f i) ⟶ X) (_ : IsColimit (Cofan.mk X g)),
    (∀ i, IsConnected (f i)) ∧ Finite ι := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{u₂, u₁} C
    X : C
    h : CategoryTheory.Limits.IsInitial X
    ⊢ Exists fun ι => Exists fun f => Exists fun g => Exists fun x => And (∀ (i :  …
  -/
  refine ⟨Empty, fun _ ↦ X, fun _ ↦ 𝟙 X, ?_⟩
  use mkCofanColimit _ (fun s ↦ IsInitial.to h s.pt) (fun s ↦ by aesop)
    (fun s m _ ↦ IsInitial.hom_ext h m _)
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{u₂, u₁} C
    X : C
    h : CategoryTheory.Limits.IsInitial X
    ⊢ And (∀ (i : Empty), CategoryTheory.PreGaloisCategory.IsConnected ((fun x =>  …
  -/
  exact ⟨by simp only [IsEmpty.forall_iff], inferInstance⟩
  /-
    🎉 no goals
  -/


private lemma has_decomp_connected_components_aux (F : C ⥤ FintypeCat.{w}) [FiberFunctor F]
    (n : ℕ) : ∀ (X : C), n = Nat.card (F.obj X) → ∃ (ι : Type) (f : ι → C)
    (g : (i : ι) → (f i) ⟶ X) (_ : IsColimit (Cofan.mk X g)),
    (∀ i, IsConnected (f i)) ∧ Finite ι := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    n : Nat
    ⊢ ∀ (X : C), Eq n (Nat.card ↑(F.obj X)) → Exists fun ι => Exists fun f => Exis …
  -/
  induction' n using Nat.strongRecOn with n hi
  /-
    case ind
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    n : Nat
    hi : ∀ (m : Nat), LT.lt m n → ∀ (X : C), Eq m (Nat.card ↑(F.obj X)) → Exists f …
    ⊢ ∀ (X : C), Eq n (Nat.card ↑(F.obj X)) → Exists fun ι => Exists fun f => Exis …
  -/
  intro X hn
  /-
    case ind
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    n : Nat
    hi : ∀ (m : Nat), LT.lt m n → ∀ (X : C), Eq m (Nat.card ↑(F.obj X)) → Exists f …
    X : C
    hn : Eq n (Nat.card ↑(F.obj X))
    ⊢ Exists fun ι => Exists fun f => Exists fun g => Exists fun x => And (∀ (i :  …
  -/
  by_cases h : IsConnected X
    /-
      case pos
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      n : Nat
      hi : ∀ (m : Nat), LT.lt m n → ∀ (X : C), Eq m (Nat.card ↑(F.obj X)) → Exists f …
      X : C
      hn : Eq n (Nat.card ↑(F.obj X))
      h : CategoryTheory.PreGaloisCategory.IsConnected X
      ⊢ Exists fun ι => Exists fun f => Exists fun g => Exists fun x => And (∀ (i :  …
    -/
  · exact has_decomp_connected_components_aux_conn X
    /-
      🎉 no goals
    -/
  /-
    case neg
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    n : Nat
    hi : ∀ (m : Nat), LT.lt m n → ∀ (X : C), Eq m (Nat.card ↑(F.obj X)) → Exists f …
    X : C
    hn : Eq n (Nat.card ↑(F.obj X))
    h : Not (CategoryTheory.PreGaloisCategory.IsConnected X)
    ⊢ Exists fun ι => Exists fun f => Exists fun g => Exists fun x => And (∀ (i :  …
  -/
  by_cases nhi : IsInitial X → False
  · obtain ⟨Y, v, hni, hvmono, hvnoiso⟩ :=
      has_non_trivial_subobject_of_not_isConnected_of_not_initial X h nhi
    /-
      case pos.intro.intro.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      n : Nat
      hi : ∀ (m : Nat), LT.lt m n → ∀ (X : C), Eq m (Nat.card ↑(F.obj X)) → Exists f …
      X : C
      hn : Eq n (Nat.card ↑(F.obj X))
      h : Not (CategoryTheory.PreGaloisCategory.IsConnected X)
      nhi : CategoryTheory.Limits.IsInitial X → False
      Y : C
      v : Quiver.Hom Y X
      hni : CategoryTheory.Limits.IsInitial Y → False
      hvmono : CategoryTheory.Mono v
      hvnoiso : Not (CategoryTheory.IsIso v)
      ⊢ Exists fun ι => Exists fun f => Exists fun g => Exists fun x => And (∀ (i :  …
    -/
    obtain ⟨Z, u, ⟨c⟩⟩ := PreGaloisCategory.monoInducesIsoOnDirectSummand v
    /-
      case pos.intro.intro.intro.intro.intro.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      n : Nat
      hi : ∀ (m : Nat), LT.lt m n → ∀ (X : C), Eq m (Nat.card ↑(F.obj X)) → Exists f …
      X : C
      hn : Eq n (Nat.card ↑(F.obj X))
      h : Not (CategoryTheory.PreGaloisCategory.IsConnected X)
      nhi : CategoryTheory.Limits.IsInitial X → False
      Y : C
      v : Quiver.Hom Y X
      hni : CategoryTheory.Limits.IsInitial Y → False
      hvmono : CategoryTheory.Mono v
      hvnoiso : Not (CategoryTheory.IsIso v)
      Z : C
      u : Quiver.Hom Z X
      c : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk v u)
      ⊢ Exists fun ι => Exists fun f => Exists fun g => Exists fun x => And (∀ (i :  …
    -/
    let t : ColimitCocone (pair Y Z) := { cocone := BinaryCofan.mk v u, isColimit := c }
    have hn1 : Nat.card (F.obj Y) < n := by
      rw [hn]
      exact lt_card_fiber_of_mono_of_notIso F v hvnoiso
    /-
      case pos.intro.intro.intro.intro.intro.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      n : Nat
      hi : ∀ (m : Nat), LT.lt m n → ∀ (X : C), Eq m (Nat.card ↑(F.obj X)) → Exists f …
      X : C
      hn : Eq n (Nat.card ↑(F.obj X))
      h : Not (CategoryTheory.PreGaloisCategory.IsConnected X)
      nhi : CategoryTheory.Limits.IsInitial X → False
      Y : C
      v : Quiver.Hom Y X
      hni : CategoryTheory.Limits.IsInitial Y → False
      hvmono : CategoryTheory.Mono v
      hvnoiso : Not (CategoryTheory.IsIso v)
      Z : C
      u : Quiver.Hom Z X
      c : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk v u)
      t : CategoryTheory.Limits.ColimitCocone (CategoryTheory.Limits.pair Y Z) := {  …
      hn1 : LT.lt (Nat.card ↑(F.obj Y)) n
      ⊢ Exists fun ι => Exists fun f => Exists fun g => Exists fun x => And (∀ (i :  …
    -/
    have i : X ≅ Y ⨿ Z := (colimit.isoColimitCocone t).symm
    have hnn : Nat.card (F.obj X) = Nat.card (F.obj Y) + Nat.card (F.obj Z) := by
      rw [card_fiber_eq_of_iso F i]
      exact card_fiber_coprod_eq_sum F Y Z
    have hn2 : Nat.card (F.obj Z) < n := by
      rw [hn, hnn, lt_add_iff_pos_left]
      exact Nat.pos_of_ne_zero (non_zero_card_fiber_of_not_initial F Y hni)
    /-
      case pos.intro.intro.intro.intro.intro.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      n : Nat
      hi : ∀ (m : Nat), LT.lt m n → ∀ (X : C), Eq m (Nat.card ↑(F.obj X)) → Exists f …
      X : C
      hn : Eq n (Nat.card ↑(F.obj X))
      h : Not (CategoryTheory.PreGaloisCategory.IsConnected X)
      nhi : CategoryTheory.Limits.IsInitial X → False
      Y : C
      v : Quiver.Hom Y X
      hni : CategoryTheory.Limits.IsInitial Y → False
      hvmono : CategoryTheory.Mono v
      hvnoiso : Not (CategoryTheory.IsIso v)
      Z : C
      u : Quiver.Hom Z X
      c : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk v u)
      t : CategoryTheory.Limits.ColimitCocone (CategoryTheory.Limits.pair Y Z) := {  …
      hn1 : LT.lt (Nat.card ↑(F.obj Y)) n
      i : CategoryTheory.Iso X (CategoryTheory.Limits.coprod Y Z)
      hnn : Eq (Nat.card ↑(F.obj X)) (HAdd.hAdd (Nat.card ↑(F.obj Y)) (Nat.card ↑(F. …
      hn2 : LT.lt (Nat.card ↑(F.obj Z)) n
      ⊢ Exists fun ι => Exists fun f => Exists fun g => Exists fun x => And (∀ (i :  …
    -/
    let ⟨ι₁, f₁, g₁, hc₁, hf₁, he₁⟩ := hi (Nat.card (F.obj Y)) hn1 Y rfl
    /-
      case pos.intro.intro.intro.intro.intro.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      n : Nat
      hi : ∀ (m : Nat), LT.lt m n → ∀ (X : C), Eq m (Nat.card ↑(F.obj X)) → Exists f …
      X : C
      hn : Eq n (Nat.card ↑(F.obj X))
      h : Not (CategoryTheory.PreGaloisCategory.IsConnected X)
      nhi : CategoryTheory.Limits.IsInitial X → False
      Y : C
      v : Quiver.Hom Y X
      hni : CategoryTheory.Limits.IsInitial Y → False
      hvmono : CategoryTheory.Mono v
      hvnoiso : Not (CategoryTheory.IsIso v)
      Z : C
      u : Quiver.Hom Z X
      c : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk v u)
      t : CategoryTheory.Limits.ColimitCocone (CategoryTheory.Limits.pair Y Z) := {  …
      hn1 : LT.lt (Nat.card ↑(F.obj Y)) n
      i : CategoryTheory.Iso X (CategoryTheory.Limits.coprod Y Z)
      hnn : Eq (Nat.card ↑(F.obj X)) (HAdd.hAdd (Nat.card ↑(F.obj Y)) (Nat.card ↑(F. …
      hn2 : LT.lt (Nat.card ↑(F.obj Z)) n
      ι₁ : Type
      f₁ : ι₁ → C
      g₁ : (i : ι₁) → Quiver.Hom (f₁ i) Y
      hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk Y g₁)
      hf₁ : ∀ (i : ι₁), CategoryTheory.PreGaloisCategory.IsConnected (f₁ i)
      he₁ : Finite ι₁
      ⊢ Exists fun ι => Exists fun f => Exists fun g => Exists fun x => And (∀ (i :  …
    -/
    let ⟨ι₂, f₂, g₂, hc₂, hf₂, he₂⟩ := hi (Nat.card (F.obj Z)) hn2 Z rfl
    refine ⟨ι₁ ⊕ ι₂, Sum.elim f₁ f₂,
      Cofan.combPairHoms (Cofan.mk Y g₁) (Cofan.mk Z g₂) (BinaryCofan.mk v u), ?_⟩
    /-
      case pos.intro.intro.intro.intro.intro.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      n : Nat
      hi : ∀ (m : Nat), LT.lt m n → ∀ (X : C), Eq m (Nat.card ↑(F.obj X)) → Exists f …
      X : C
      hn : Eq n (Nat.card ↑(F.obj X))
      h : Not (CategoryTheory.PreGaloisCategory.IsConnected X)
      nhi : CategoryTheory.Limits.IsInitial X → False
      Y : C
      v : Quiver.Hom Y X
      hni : CategoryTheory.Limits.IsInitial Y → False
      hvmono : CategoryTheory.Mono v
      hvnoiso : Not (CategoryTheory.IsIso v)
      Z : C
      u : Quiver.Hom Z X
      c : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk v u)
      t : CategoryTheory.Limits.ColimitCocone (CategoryTheory.Limits.pair Y Z) := {  …
      hn1 : LT.lt (Nat.card ↑(F.obj Y)) n
      i : CategoryTheory.Iso X (CategoryTheory.Limits.coprod Y Z)
      hnn : Eq (Nat.card ↑(F.obj X)) (HAdd.hAdd (Nat.card ↑(F.obj Y)) (Nat.card ↑(F. …
      hn2 : LT.lt (Nat.card ↑(F.obj Z)) n
      ι₁ : Type
      f₁ : ι₁ → C
      g₁ : (i : ι₁) → Quiver.Hom (f₁ i) Y
      hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk Y g₁)
      hf₁ : ∀ (i : ι₁), CategoryTheory.PreGaloisCategory.IsConnected (f₁ i)
      he₁ : Finite ι₁
      ι₂ : Type
      f₂ : ι₂ → C
      g₂ : (i : ι₂) → Quiver.Hom (f₂ i) Z
      hc₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk Z g₂)
      hf₂ : ∀ (i : ι₂), CategoryTheory.PreGaloisCategory.IsConnected (f₂ i)
      he₂ : Finite ι₂
      ⊢ Exists fun x => And (∀ (i : Sum ι₁ ι₂), CategoryTheory.PreGaloisCategory.IsC …
    -/
    use Cofan.combPairIsColimit hc₁ hc₂ c
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      n : Nat
      hi : ∀ (m : Nat), LT.lt m n → ∀ (X : C), Eq m (Nat.card ↑(F.obj X)) → Exists f …
      X : C
      hn : Eq n (Nat.card ↑(F.obj X))
      h : Not (CategoryTheory.PreGaloisCategory.IsConnected X)
      nhi : CategoryTheory.Limits.IsInitial X → False
      Y : C
      v : Quiver.Hom Y X
      hni : CategoryTheory.Limits.IsInitial Y → False
      hvmono : CategoryTheory.Mono v
      hvnoiso : Not (CategoryTheory.IsIso v)
      Z : C
      u : Quiver.Hom Z X
      c : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk v u)
      t : CategoryTheory.Limits.ColimitCocone (CategoryTheory.Limits.pair Y Z) := {  …
      hn1 : LT.lt (Nat.card ↑(F.obj Y)) n
      i : CategoryTheory.Iso X (CategoryTheory.Limits.coprod Y Z)
      hnn : Eq (Nat.card ↑(F.obj X)) (HAdd.hAdd (Nat.card ↑(F.obj Y)) (Nat.card ↑(F. …
      hn2 : LT.lt (Nat.card ↑(F.obj Z)) n
      ι₁ : Type
      f₁ : ι₁ → C
      g₁ : (i : ι₁) → Quiver.Hom (f₁ i) Y
      hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk Y g₁)
      hf₁ : ∀ (i : ι₁), CategoryTheory.PreGaloisCategory.IsConnected (f₁ i)
      he₁ : Finite ι₁
      ι₂ : Type
      f₂ : ι₂ → C
      g₂ : (i : ι₂) → Quiver.Hom (f₂ i) Z
      hc₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk Z g₂)
      hf₂ : ∀ (i : ι₂), CategoryTheory.PreGaloisCategory.IsConnected (f₂ i)
      he₂ : Finite ι₂
      ⊢ And (∀ (i : Sum ι₁ ι₂), CategoryTheory.PreGaloisCategory.IsConnected (Sum.el …
    -/
    refine ⟨fun i ↦ ?_, inferInstance⟩
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      n : Nat
      hi : ∀ (m : Nat), LT.lt m n → ∀ (X : C), Eq m (Nat.card ↑(F.obj X)) → Exists f …
      X : C
      hn : Eq n (Nat.card ↑(F.obj X))
      h : Not (CategoryTheory.PreGaloisCategory.IsConnected X)
      nhi : CategoryTheory.Limits.IsInitial X → False
      Y : C
      v : Quiver.Hom Y X
      hni : CategoryTheory.Limits.IsInitial Y → False
      hvmono : CategoryTheory.Mono v
      hvnoiso : Not (CategoryTheory.IsIso v)
      Z : C
      u : Quiver.Hom Z X
      c : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk v u)
      t : CategoryTheory.Limits.ColimitCocone (CategoryTheory.Limits.pair Y Z) := {  …
      hn1 : LT.lt (Nat.card ↑(F.obj Y)) n
      i✝ : CategoryTheory.Iso X (CategoryTheory.Limits.coprod Y Z)
      hnn : Eq (Nat.card ↑(F.obj X)) (HAdd.hAdd (Nat.card ↑(F.obj Y)) (Nat.card ↑(F. …
      hn2 : LT.lt (Nat.card ↑(F.obj Z)) n
      ι₁ : Type
      f₁ : ι₁ → C
      g₁ : (i : ι₁) → Quiver.Hom (f₁ i) Y
      hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk Y g₁)
      hf₁ : ∀ (i : ι₁), CategoryTheory.PreGaloisCategory.IsConnected (f₁ i)
      he₁ : Finite ι₁
      ι₂ : Type
      f₂ : ι₂ → C
      g₂ : (i : ι₂) → Quiver.Hom (f₂ i) Z
      hc₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk Z g₂)
      hf₂ : ∀ (i : ι₂), CategoryTheory.PreGaloisCategory.IsConnected (f₂ i)
      he₂ : Finite ι₂
      i : Sum ι₁ ι₂
      ⊢ CategoryTheory.PreGaloisCategory.IsConnected (Sum.elim f₁ f₂ i)
    -/
    cases i
      /-
        case h.inl
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        inst✝¹ : CategoryTheory.GaloisCategory C
        F : CategoryTheory.Functor C FintypeCat
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        n : Nat
        hi : ∀ (m : Nat), LT.lt m n → ∀ (X : C), Eq m (Nat.card ↑(F.obj X)) → Exists f …
        X : C
        hn : Eq n (Nat.card ↑(F.obj X))
        h : Not (CategoryTheory.PreGaloisCategory.IsConnected X)
        nhi : CategoryTheory.Limits.IsInitial X → False
        Y : C
        v : Quiver.Hom Y X
        hni : CategoryTheory.Limits.IsInitial Y → False
        hvmono : CategoryTheory.Mono v
        hvnoiso : Not (CategoryTheory.IsIso v)
        Z : C
        u : Quiver.Hom Z X
        c : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk v u)
        t : CategoryTheory.Limits.ColimitCocone (CategoryTheory.Limits.pair Y Z) := {  …
        hn1 : LT.lt (Nat.card ↑(F.obj Y)) n
        i : CategoryTheory.Iso X (CategoryTheory.Limits.coprod Y Z)
        hnn : Eq (Nat.card ↑(F.obj X)) (HAdd.hAdd (Nat.card ↑(F.obj Y)) (Nat.card ↑(F. …
        hn2 : LT.lt (Nat.card ↑(F.obj Z)) n
        ι₁ : Type
        f₁ : ι₁ → C
        g₁ : (i : ι₁) → Quiver.Hom (f₁ i) Y
        hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk Y g₁)
        hf₁ : ∀ (i : ι₁), CategoryTheory.PreGaloisCategory.IsConnected (f₁ i)
        he₁ : Finite ι₁
        ι₂ : Type
        f₂ : ι₂ → C
        g₂ : (i : ι₂) → Quiver.Hom (f₂ i) Z
        hc₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk Z g₂)
        hf₂ : ∀ (i : ι₂), CategoryTheory.PreGaloisCategory.IsConnected (f₂ i)
        he₂ : Finite ι₂
        val✝ : ι₁
        ⊢ CategoryTheory.PreGaloisCategory.IsConnected (Sum.elim f₁ f₂ (Sum.inl val✝))
      -/
    · exact hf₁ _
      /-
        🎉 no goals
      -/
      /-
        case h.inr
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        inst✝¹ : CategoryTheory.GaloisCategory C
        F : CategoryTheory.Functor C FintypeCat
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        n : Nat
        hi : ∀ (m : Nat), LT.lt m n → ∀ (X : C), Eq m (Nat.card ↑(F.obj X)) → Exists f …
        X : C
        hn : Eq n (Nat.card ↑(F.obj X))
        h : Not (CategoryTheory.PreGaloisCategory.IsConnected X)
        nhi : CategoryTheory.Limits.IsInitial X → False
        Y : C
        v : Quiver.Hom Y X
        hni : CategoryTheory.Limits.IsInitial Y → False
        hvmono : CategoryTheory.Mono v
        hvnoiso : Not (CategoryTheory.IsIso v)
        Z : C
        u : Quiver.Hom Z X
        c : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk v u)
        t : CategoryTheory.Limits.ColimitCocone (CategoryTheory.Limits.pair Y Z) := {  …
        hn1 : LT.lt (Nat.card ↑(F.obj Y)) n
        i : CategoryTheory.Iso X (CategoryTheory.Limits.coprod Y Z)
        hnn : Eq (Nat.card ↑(F.obj X)) (HAdd.hAdd (Nat.card ↑(F.obj Y)) (Nat.card ↑(F. …
        hn2 : LT.lt (Nat.card ↑(F.obj Z)) n
        ι₁ : Type
        f₁ : ι₁ → C
        g₁ : (i : ι₁) → Quiver.Hom (f₁ i) Y
        hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk Y g₁)
        hf₁ : ∀ (i : ι₁), CategoryTheory.PreGaloisCategory.IsConnected (f₁ i)
        he₁ : Finite ι₁
        ι₂ : Type
        f₂ : ι₂ → C
        g₂ : (i : ι₂) → Quiver.Hom (f₂ i) Z
        hc₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk Z g₂)
        hf₂ : ∀ (i : ι₂), CategoryTheory.PreGaloisCategory.IsConnected (f₂ i)
        he₂ : Finite ι₂
        val✝ : ι₂
        ⊢ CategoryTheory.PreGaloisCategory.IsConnected (Sum.elim f₁ f₂ (Sum.inr val✝))
      -/
    · exact hf₂ _
      /-
        🎉 no goals
      -/
    /-
      case neg
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      n : Nat
      hi : ∀ (m : Nat), LT.lt m n → ∀ (X : C), Eq m (Nat.card ↑(F.obj X)) → Exists f …
      X : C
      hn : Eq n (Nat.card ↑(F.obj X))
      h : Not (CategoryTheory.PreGaloisCategory.IsConnected X)
      nhi : Not (CategoryTheory.Limits.IsInitial X → False)
      ⊢ Exists fun ι => Exists fun f => Exists fun g => Exists fun x => And (∀ (i :  …
    -/
  · simp only [not_forall, not_false_eq_true] at nhi
    /-
      case neg
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      n : Nat
      hi : ∀ (m : Nat), LT.lt m n → ∀ (X : C), Eq m (Nat.card ↑(F.obj X)) → Exists f …
      X : C
      hn : Eq n (Nat.card ↑(F.obj X))
      h : Not (CategoryTheory.PreGaloisCategory.IsConnected X)
      nhi : Exists fun x => True
      ⊢ Exists fun ι => Exists fun f => Exists fun g => Exists fun x => And (∀ (i :  …
    -/
    obtain ⟨hi⟩ := nhi
    /-
      case neg.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      n : Nat
      hi✝ : ∀ (m : Nat), LT.lt m n → ∀ (X : C), Eq m (Nat.card ↑(F.obj X)) → Exists  …
      X : C
      hn : Eq n (Nat.card ↑(F.obj X))
      h : Not (CategoryTheory.PreGaloisCategory.IsConnected X)
      hi : CategoryTheory.Limits.IsInitial X
      h✝ : True
      ⊢ Exists fun ι => Exists fun f => Exists fun g => Exists fun x => And (∀ (i :  …
    -/
    exact has_decomp_connected_components_aux_initial X hi
    /-
      🎉 no goals
    -/


/-- In a Galois category, every object is the sum of connected objects. -/
theorem has_decomp_connected_components (X : C) :
    ∃ (ι : Type) (f : ι → C) (g : (i : ι) → f i ⟶ X) (_ : IsColimit (Cofan.mk X g)),
      (∀ i, IsConnected (f i)) ∧ Finite ι := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
    inst✝ : CategoryTheory.GaloisCategory C
    X : C
    ⊢ Exists fun ι => Exists fun f => Exists fun g => Exists fun x => And (∀ (i :  …
  -/
  let F := GaloisCategory.getFiberFunctor C
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
    inst✝ : CategoryTheory.GaloisCategory C
    X : C
    F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
    ⊢ Exists fun ι => Exists fun f => Exists fun g => Exists fun x => And (∀ (i :  …
  -/
  exact has_decomp_connected_components_aux F (Nat.card <| F.obj X) X rfl
  /-
    🎉 no goals
  -/


/-- In a Galois category, every object is the sum of connected objects. -/
theorem has_decomp_connected_components' (X : C) :
    ∃ (ι : Type) (_ : Finite ι) (f : ι → C) (_ : ∐ f ≅ X), ∀ i, IsConnected (f i) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
    inst✝ : CategoryTheory.GaloisCategory C
    X : C
    ⊢ Exists fun ι => Exists fun x => Exists fun f => Exists fun x => ∀ (i : ι), C …
  -/
  obtain ⟨ι, f, g, hl, hc, hf⟩ := has_decomp_connected_components X
  /-
    case intro.intro.intro.intro.intro
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
    inst✝ : CategoryTheory.GaloisCategory C
    X : C
    ι : Type
    f : ι → C
    g : (i : ι) → Quiver.Hom (f i) X
    hl : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk X g)
    hc : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (f i)
    hf : Finite ι
    ⊢ Exists fun ι => Exists fun x => Exists fun f => Exists fun x => ∀ (i : ι), C …
  -/
  exact ⟨ι, hf, f, colimit.isoColimitCocone ⟨Cofan.mk X g, hl⟩, hc⟩
  /-
    🎉 no goals
  -/


/-- Every element in the fiber of `X` lies in some connected component of `X`. -/
lemma fiber_in_connected_component (X : C) (x : F.obj X) : ∃ (Y : C) (i : Y ⟶ X) (y : F.obj Y),
    F.map i y = x ∧ IsConnected Y ∧ Mono i := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    x : ↑(F.obj X)
    ⊢ Exists fun Y => Exists fun i => Exists fun y => And (Eq (F.map i y) x) (And  …
  -/
  obtain ⟨ι, f, g, hl, hc, he⟩ := has_decomp_connected_components X
  /-
    case intro.intro.intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    x : ↑(F.obj X)
    ι : Type
    f : ι → C
    g : (i : ι) → Quiver.Hom (f i) X
    hl : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk X g)
    hc : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (f i)
    he : Finite ι
    ⊢ Exists fun Y => Exists fun i => Exists fun y => And (Eq (F.map i y) x) (And  …
  -/
  have : Fintype ι := Fintype.ofFinite ι
  /-
    case intro.intro.intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    x : ↑(F.obj X)
    ι : Type
    f : ι → C
    g : (i : ι) → Quiver.Hom (f i) X
    hl : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk X g)
    hc : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (f i)
    he : Finite ι
    this : Fintype ι
    ⊢ Exists fun Y => Exists fun i => Exists fun y => And (Eq (F.map i y) x) (And  …
  -/
  let s : Cocone (Discrete.functor f ⋙ F) := F.mapCocone (Cofan.mk X g)
  /-
    case intro.intro.intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    x : ↑(F.obj X)
    ι : Type
    f : ι → C
    g : (i : ι) → Quiver.Hom (f i) X
    hl : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk X g)
    hc : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (f i)
    he : Finite ι
    this : Fintype ι
    s : CategoryTheory.Limits.Cocone ((CategoryTheory.Discrete.functor f).comp F)  …
    ⊢ Exists fun Y => Exists fun i => Exists fun y => And (Eq (F.map i y) x) (And  …
  -/
  let s' : IsColimit s := isColimitOfPreserves F hl
  /-
    case intro.intro.intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    x : ↑(F.obj X)
    ι : Type
    f : ι → C
    g : (i : ι) → Quiver.Hom (f i) X
    hl : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk X g)
    hc : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (f i)
    he : Finite ι
    this : Fintype ι
    s : CategoryTheory.Limits.Cocone ((CategoryTheory.Discrete.functor f).comp F)  …
    s' : CategoryTheory.Limits.IsColimit s := CategoryTheory.Limits.isColimitOfPre …
    ⊢ Exists fun Y => Exists fun i => Exists fun y => And (Eq (F.map i y) x) (And  …
  -/
  obtain ⟨⟨j⟩, z, h⟩ := Concrete.isColimit_exists_rep _ s' x
  /-
    case intro.intro.intro.intro.intro.intro.mk.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    x : ↑(F.obj X)
    ι : Type
    f : ι → C
    g : (i : ι) → Quiver.Hom (f i) X
    hl : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk X g)
    hc : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (f i)
    he : Finite ι
    this : Fintype ι
    s : CategoryTheory.Limits.Cocone ((CategoryTheory.Discrete.functor f).comp F)  …
    s' : CategoryTheory.Limits.IsColimit s := CategoryTheory.Limits.isColimitOfPre …
    j : ι
    z : (CategoryTheory.forget FintypeCat).obj (((CategoryTheory.Discrete.functor  …
    h : Eq ((s.ι.app { as := j }) z) x
    ⊢ Exists fun Y => Exists fun i => Exists fun y => And (Eq (F.map i y) x) (And  …
  -/
  refine ⟨f j, g j, z, ⟨?_, hc j, MonoCoprod.mono_inj _ (Cofan.mk X g) hl j⟩⟩
  /-
    case intro.intro.intro.intro.intro.intro.mk.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    x : ↑(F.obj X)
    ι : Type
    f : ι → C
    g : (i : ι) → Quiver.Hom (f i) X
    hl : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk X g)
    hc : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (f i)
    he : Finite ι
    this : Fintype ι
    s : CategoryTheory.Limits.Cocone ((CategoryTheory.Discrete.functor f).comp F)  …
    s' : CategoryTheory.Limits.IsColimit s := CategoryTheory.Limits.isColimitOfPre …
    j : ι
    z : (CategoryTheory.forget FintypeCat).obj (((CategoryTheory.Discrete.functor  …
    h : Eq ((s.ι.app { as := j }) z) x
    ⊢ Eq (F.map (g j) z) x
  -/
  subst h
  /-
    case intro.intro.intro.intro.intro.intro.mk.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    ι : Type
    f : ι → C
    g : (i : ι) → Quiver.Hom (f i) X
    hl : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk X g)
    hc : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (f i)
    he : Finite ι
    this : Fintype ι
    s : CategoryTheory.Limits.Cocone ((CategoryTheory.Discrete.functor f).comp F)  …
    s' : CategoryTheory.Limits.IsColimit s := CategoryTheory.Limits.isColimitOfPre …
    j : ι
    z : (CategoryTheory.forget FintypeCat).obj (((CategoryTheory.Discrete.functor  …
    ⊢ Eq (F.map (g j) z) ((s.ι.app { as := j }) z)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Up to isomorphism an element of the fiber of `X` only lies in one connected component. -/
lemma connected_component_unique {X A B : C} [IsConnected A] [IsConnected B] (a : F.obj A)
    (b : F.obj B) (i : A ⟶ X) (j : B ⟶ X) (h : F.map i a = F.map j b) [Mono i] [Mono j] :
    ∃ (f : A ≅ B), F.map f.hom a = b := by
  /- We consider the fiber product of A and B over X. This is a non-empty (because of `h`)
  subobject of `A` and `B` and hence isomorphic to `A` and `B` by connectedness. -/
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
    inst✝⁵ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A B : C
    inst✝³ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝² : CategoryTheory.PreGaloisCategory.IsConnected B
    a : ↑(F.obj A)
    b : ↑(F.obj B)
    i : Quiver.Hom A X
    j : Quiver.Hom B X
    h : Eq (F.map i a) (F.map j b)
    inst✝¹ : CategoryTheory.Mono i
    inst✝ : CategoryTheory.Mono j
    ⊢ Exists fun f => Eq (F.map f.hom a) b
  -/
  let Y : C := pullback i j
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
    inst✝⁵ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A B : C
    inst✝³ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝² : CategoryTheory.PreGaloisCategory.IsConnected B
    a : ↑(F.obj A)
    b : ↑(F.obj B)
    i : Quiver.Hom A X
    j : Quiver.Hom B X
    h : Eq (F.map i a) (F.map j b)
    inst✝¹ : CategoryTheory.Mono i
    inst✝ : CategoryTheory.Mono j
    Y : C := CategoryTheory.Limits.pullback i j
    ⊢ Exists fun f => Eq (F.map f.hom a) b
  -/
  let u : Y ⟶ A := pullback.fst i j
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
    inst✝⁵ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A B : C
    inst✝³ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝² : CategoryTheory.PreGaloisCategory.IsConnected B
    a : ↑(F.obj A)
    b : ↑(F.obj B)
    i : Quiver.Hom A X
    j : Quiver.Hom B X
    h : Eq (F.map i a) (F.map j b)
    inst✝¹ : CategoryTheory.Mono i
    inst✝ : CategoryTheory.Mono j
    Y : C := CategoryTheory.Limits.pullback i j
    u : Quiver.Hom Y A := CategoryTheory.Limits.pullback.fst i j
    ⊢ Exists fun f => Eq (F.map f.hom a) b
  -/
  let v : Y ⟶ B := pullback.snd i j
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
    inst✝⁵ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A B : C
    inst✝³ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝² : CategoryTheory.PreGaloisCategory.IsConnected B
    a : ↑(F.obj A)
    b : ↑(F.obj B)
    i : Quiver.Hom A X
    j : Quiver.Hom B X
    h : Eq (F.map i a) (F.map j b)
    inst✝¹ : CategoryTheory.Mono i
    inst✝ : CategoryTheory.Mono j
    Y : C := CategoryTheory.Limits.pullback i j
    u : Quiver.Hom Y A := CategoryTheory.Limits.pullback.fst i j
    v : Quiver.Hom Y B := CategoryTheory.Limits.pullback.snd i j
    ⊢ Exists fun f => Eq (F.map f.hom a) b
  -/
  let G := F ⋙ FintypeCat.incl
  let e : F.obj Y ≃ { p : F.obj A × F.obj B // F.map i p.1 = F.map j p.2 } :=
    fiberPullbackEquiv F i j
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
    inst✝⁵ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A B : C
    inst✝³ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝² : CategoryTheory.PreGaloisCategory.IsConnected B
    a : ↑(F.obj A)
    b : ↑(F.obj B)
    i : Quiver.Hom A X
    j : Quiver.Hom B X
    h : Eq (F.map i a) (F.map j b)
    inst✝¹ : CategoryTheory.Mono i
    inst✝ : CategoryTheory.Mono j
    Y : C := CategoryTheory.Limits.pullback i j
    u : Quiver.Hom Y A := CategoryTheory.Limits.pullback.fst i j
    v : Quiver.Hom Y B := CategoryTheory.Limits.pullback.snd i j
    G : CategoryTheory.Functor C (Type w) := F.comp FintypeCat.incl
    e : Equiv (↑(F.obj Y)) (Subtype fun p => Eq (F.map i p.1) (F.map j p.2)) := Ca …
    ⊢ Exists fun f => Eq (F.map f.hom a) b
  -/
  let y : F.obj Y := e.symm ⟨(a, b), h⟩
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
    inst✝⁵ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A B : C
    inst✝³ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝² : CategoryTheory.PreGaloisCategory.IsConnected B
    a : ↑(F.obj A)
    b : ↑(F.obj B)
    i : Quiver.Hom A X
    j : Quiver.Hom B X
    h : Eq (F.map i a) (F.map j b)
    inst✝¹ : CategoryTheory.Mono i
    inst✝ : CategoryTheory.Mono j
    Y : C := CategoryTheory.Limits.pullback i j
    u : Quiver.Hom Y A := CategoryTheory.Limits.pullback.fst i j
    v : Quiver.Hom Y B := CategoryTheory.Limits.pullback.snd i j
    G : CategoryTheory.Functor C (Type w) := F.comp FintypeCat.incl
    e : Equiv (↑(F.obj Y)) (Subtype fun p => Eq (F.map i p.1) (F.map j p.2)) := Ca …
    y : ↑(F.obj Y) := e.symm ⟨{ fst := a, snd := b }, h⟩
    ⊢ Exists fun f => Eq (F.map f.hom a) b
  -/
  have hn : IsInitial Y → False := not_initial_of_inhabited F y
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
    inst✝⁵ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A B : C
    inst✝³ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝² : CategoryTheory.PreGaloisCategory.IsConnected B
    a : ↑(F.obj A)
    b : ↑(F.obj B)
    i : Quiver.Hom A X
    j : Quiver.Hom B X
    h : Eq (F.map i a) (F.map j b)
    inst✝¹ : CategoryTheory.Mono i
    inst✝ : CategoryTheory.Mono j
    Y : C := CategoryTheory.Limits.pullback i j
    u : Quiver.Hom Y A := CategoryTheory.Limits.pullback.fst i j
    v : Quiver.Hom Y B := CategoryTheory.Limits.pullback.snd i j
    G : CategoryTheory.Functor C (Type w) := F.comp FintypeCat.incl
    e : Equiv (↑(F.obj Y)) (Subtype fun p => Eq (F.map i p.1) (F.map j p.2)) := Ca …
    y : ↑(F.obj Y) := e.symm ⟨{ fst := a, snd := b }, h⟩
    hn : CategoryTheory.Limits.IsInitial Y → False
    ⊢ Exists fun f => Eq (F.map f.hom a) b
  -/
  have : IsIso u := IsConnected.noTrivialComponent Y u hn
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
    inst✝⁵ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A B : C
    inst✝³ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝² : CategoryTheory.PreGaloisCategory.IsConnected B
    a : ↑(F.obj A)
    b : ↑(F.obj B)
    i : Quiver.Hom A X
    j : Quiver.Hom B X
    h : Eq (F.map i a) (F.map j b)
    inst✝¹ : CategoryTheory.Mono i
    inst✝ : CategoryTheory.Mono j
    Y : C := CategoryTheory.Limits.pullback i j
    u : Quiver.Hom Y A := CategoryTheory.Limits.pullback.fst i j
    v : Quiver.Hom Y B := CategoryTheory.Limits.pullback.snd i j
    G : CategoryTheory.Functor C (Type w) := F.comp FintypeCat.incl
    e : Equiv (↑(F.obj Y)) (Subtype fun p => Eq (F.map i p.1) (F.map j p.2)) := Ca …
    y : ↑(F.obj Y) := e.symm ⟨{ fst := a, snd := b }, h⟩
    hn : CategoryTheory.Limits.IsInitial Y → False
    this : CategoryTheory.IsIso u
    ⊢ Exists fun f => Eq (F.map f.hom a) b
  -/
  have : IsIso v := IsConnected.noTrivialComponent Y v hn
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
    inst✝⁵ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A B : C
    inst✝³ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝² : CategoryTheory.PreGaloisCategory.IsConnected B
    a : ↑(F.obj A)
    b : ↑(F.obj B)
    i : Quiver.Hom A X
    j : Quiver.Hom B X
    h : Eq (F.map i a) (F.map j b)
    inst✝¹ : CategoryTheory.Mono i
    inst✝ : CategoryTheory.Mono j
    Y : C := CategoryTheory.Limits.pullback i j
    u : Quiver.Hom Y A := CategoryTheory.Limits.pullback.fst i j
    v : Quiver.Hom Y B := CategoryTheory.Limits.pullback.snd i j
    G : CategoryTheory.Functor C (Type w) := F.comp FintypeCat.incl
    e : Equiv (↑(F.obj Y)) (Subtype fun p => Eq (F.map i p.1) (F.map j p.2)) := Ca …
    y : ↑(F.obj Y) := e.symm ⟨{ fst := a, snd := b }, h⟩
    hn : CategoryTheory.Limits.IsInitial Y → False
    this✝ : CategoryTheory.IsIso u
    this : CategoryTheory.IsIso v
    ⊢ Exists fun f => Eq (F.map f.hom a) b
  -/
  use (asIso u).symm ≪≫ asIso v
  have hu : G.map u y = a := by
    simp only [y, e, ← PreservesPullback.iso_hom_fst G, fiberPullbackEquiv, Iso.toEquiv_comp,
      Equiv.symm_trans_apply, Iso.toEquiv_symm_fun, types_comp_apply, inv_hom_id_apply]
    erw [Types.pullbackIsoPullback_inv_fst_apply (F.map i) (F.map j)]
  have hv : G.map v y = b := by
    simp only [y, e, ← PreservesPullback.iso_hom_snd G, fiberPullbackEquiv, Iso.toEquiv_comp,
      Equiv.symm_trans_apply, Iso.toEquiv_symm_fun, types_comp_apply, inv_hom_id_apply]
    erw [Types.pullbackIsoPullback_inv_snd_apply (F.map i) (F.map j)]
  /-
    case h
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
    inst✝⁵ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A B : C
    inst✝³ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝² : CategoryTheory.PreGaloisCategory.IsConnected B
    a : ↑(F.obj A)
    b : ↑(F.obj B)
    i : Quiver.Hom A X
    j : Quiver.Hom B X
    h : Eq (F.map i a) (F.map j b)
    inst✝¹ : CategoryTheory.Mono i
    inst✝ : CategoryTheory.Mono j
    Y : C := CategoryTheory.Limits.pullback i j
    u : Quiver.Hom Y A := CategoryTheory.Limits.pullback.fst i j
    v : Quiver.Hom Y B := CategoryTheory.Limits.pullback.snd i j
    G : CategoryTheory.Functor C (Type w) := F.comp FintypeCat.incl
    e : Equiv (↑(F.obj Y)) (Subtype fun p => Eq (F.map i p.1) (F.map j p.2)) := Ca …
    y : ↑(F.obj Y) := e.symm ⟨{ fst := a, snd := b }, h⟩
    hn : CategoryTheory.Limits.IsInitial Y → False
    this✝ : CategoryTheory.IsIso u
    this : CategoryTheory.IsIso v
    hu : Eq (G.map u y) a
    hv : Eq (G.map v y) b
    ⊢ Eq (F.map ((CategoryTheory.asIso u).symm.trans (CategoryTheory.asIso v)).hom …
  -/
  rw [← hu, ← hv]
  /-
    case h
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
    inst✝⁵ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A B : C
    inst✝³ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝² : CategoryTheory.PreGaloisCategory.IsConnected B
    a : ↑(F.obj A)
    b : ↑(F.obj B)
    i : Quiver.Hom A X
    j : Quiver.Hom B X
    h : Eq (F.map i a) (F.map j b)
    inst✝¹ : CategoryTheory.Mono i
    inst✝ : CategoryTheory.Mono j
    Y : C := CategoryTheory.Limits.pullback i j
    u : Quiver.Hom Y A := CategoryTheory.Limits.pullback.fst i j
    v : Quiver.Hom Y B := CategoryTheory.Limits.pullback.snd i j
    G : CategoryTheory.Functor C (Type w) := F.comp FintypeCat.incl
    e : Equiv (↑(F.obj Y)) (Subtype fun p => Eq (F.map i p.1) (F.map j p.2)) := Ca …
    y : ↑(F.obj Y) := e.symm ⟨{ fst := a, snd := b }, h⟩
    hn : CategoryTheory.Limits.IsInitial Y → False
    this✝ : CategoryTheory.IsIso u
    this : CategoryTheory.IsIso v
    hu : Eq (G.map u y) a
    hv : Eq (G.map v y) b
    ⊢ Eq (F.map ((CategoryTheory.asIso u).symm.trans (CategoryTheory.asIso v)).hom …
  -/
  show (F.toPrefunctor.map u ≫ F.toPrefunctor.map _) y = F.toPrefunctor.map v y
  simp only [← F.map_comp, Iso.trans_hom, Iso.symm_hom, asIso_inv, asIso_hom,
    IsIso.hom_inv_id_assoc]


/-- The self product of `X` indexed by its fiber. -/
@[simp]
private noncomputable def selfProd : C := ∏ᶜ (fun _ : F.obj X ↦ X)


/-- For `g : F.obj X → F.obj X`, this is the element in the fiber of the self product,
which has at index `x : F.obj X` the element `g x`. -/
private noncomputable def mkSelfProdFib : F.obj (selfProd F X) :=
  (PreservesProduct.iso F _).inv ((Concrete.productEquiv (fun _ : F.obj X ↦ F.obj X)).symm id)


@[simp]
private lemma mkSelfProdFib_map_π (t : F.obj X) : F.map (Pi.π _ t) (mkSelfProdFib F X) = t := by
  rw [← congrFun (piComparison_comp_π F _ t), FintypeCat.comp_apply,
    ← PreservesProduct.iso_hom]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    t : ↑(F.obj X)
    ⊢ Eq (CategoryTheory.Limits.Pi.π (fun b => F.obj X) t ((CategoryTheory.Limits. …
  -/
  simp only [mkSelfProdFib, FintypeCat.inv_hom_id_apply]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    t : ↑(F.obj X)
    ⊢ Eq (CategoryTheory.Limits.Pi.π (fun b => F.obj X) t ((CategoryTheory.Limits. …
  -/
  exact Concrete.productEquiv_symm_apply_π.{w, w, w+1} (fun _ : F.obj X ↦ F.obj X) id t
  /-
    🎉 no goals
  -/


/-- For each `x : F.obj X`, this is the composition of `u` with the projection at `x`. -/
@[simp]
private noncomputable def selfProdProj (x : F.obj X) : A ⟶ X := u ≫ Pi.π _ x


private lemma selfProdProj_fiber (x : F.obj X) :
    F.map (selfProdProj u x) a = x := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
    a : ↑(F.obj A)
    h : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
    x : ↑(F.obj X)
    ⊢ Eq (F.map (CategoryTheory.PreGaloisCategory.selfProdProj u x) a) x
  -/
  simp only [selfProdProj, selfProd, F.map_comp, FintypeCat.comp_apply, h]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
    a : ↑(F.obj A)
    h : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
    x : ↑(F.obj X)
    ⊢ Eq (F.map (CategoryTheory.Limits.Pi.π (fun x => X) x) (CategoryTheory.PreGal …
  -/
  rw [mkSelfProdFib_map_π F X x]
  /-
    🎉 no goals
  -/


/-- An element `b : F.obj A` defines a permutation of the fiber of `X` by projecting onto the
`F.map u b` factor. -/
private noncomputable def fiberPerm (b : F.obj A) : F.obj X ≃ F.obj X := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
    a : ↑(F.obj A)
    h : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    b : ↑(F.obj A)
    ⊢ Equiv ↑(F.obj X) ↑(F.obj X)
  -/
  let σ (t : F.obj X) : F.obj X := F.map (selfProdProj u t) b
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
    a : ↑(F.obj A)
    h : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    b : ↑(F.obj A)
    σ : ↑(F.obj X) → ↑(F.obj X) := fun t => F.map (CategoryTheory.PreGaloisCategor …
    ⊢ Equiv ↑(F.obj X) ↑(F.obj X)
  -/
  apply Equiv.ofBijective σ
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
    a : ↑(F.obj A)
    h : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    b : ↑(F.obj A)
    σ : ↑(F.obj X) → ↑(F.obj X) := fun t => F.map (CategoryTheory.PreGaloisCategor …
    ⊢ Function.Bijective σ
  -/
  apply Finite.injective_iff_bijective.mp
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
    a : ↑(F.obj A)
    h : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    b : ↑(F.obj A)
    σ : ↑(F.obj X) → ↑(F.obj X) := fun t => F.map (CategoryTheory.PreGaloisCategor …
    ⊢ Function.Injective σ
  -/
  intro t s (hs : F.map (selfProdProj u t) b = F.map (selfProdProj u s) b)
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
    a : ↑(F.obj A)
    h : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    b : ↑(F.obj A)
    σ : ↑(F.obj X) → ↑(F.obj X) := fun t => F.map (CategoryTheory.PreGaloisCategor …
    t s : ↑(F.obj X)
    hs : Eq (F.map (CategoryTheory.PreGaloisCategory.selfProdProj u t) b) (F.map ( …
    ⊢ Eq t s
  -/
  show id t = id s
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
    a : ↑(F.obj A)
    h : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    b : ↑(F.obj A)
    σ : ↑(F.obj X) → ↑(F.obj X) := fun t => F.map (CategoryTheory.PreGaloisCategor …
    t s : ↑(F.obj X)
    hs : Eq (F.map (CategoryTheory.PreGaloisCategory.selfProdProj u t) b) (F.map ( …
    ⊢ Eq (id t) (id s)
  -/
  have h' : selfProdProj u t = selfProdProj u s := evaluation_injective_of_isConnected F A X b hs
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
    a : ↑(F.obj A)
    h : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    b : ↑(F.obj A)
    σ : ↑(F.obj X) → ↑(F.obj X) := fun t => F.map (CategoryTheory.PreGaloisCategor …
    t s : ↑(F.obj X)
    hs : Eq (F.map (CategoryTheory.PreGaloisCategory.selfProdProj u t) b) (F.map ( …
    h' : Eq (CategoryTheory.PreGaloisCategory.selfProdProj u t) (CategoryTheory.Pr …
    ⊢ Eq (id t) (id s)
  -/
  rw [← selfProdProj_fiber h s, ← selfProdProj_fiber h t, h']
  /-
    🎉 no goals
  -/


/-- Twisting `u` by `fiberPerm h b` yields an inclusion of `A` into `selfProd F X`. -/
private noncomputable def selfProdPermIncl (b : F.obj A) : A ⟶ selfProd F X :=
  u ≫ (Pi.whiskerEquiv (fiberPerm h b) (fun _ => Iso.refl X)).inv


private instance [Mono u] (b : F.obj A) : Mono (selfProdPermIncl h b) := mono_comp _ _


/-- Key technical lemma: the twisted inclusion `selfProdPermIncl h b` maps `a` to `F.map u b`. -/
private lemma selfProdTermIncl_fib_eq (b : F.obj A) :
    F.map u b = F.map (selfProdPermIncl h b) a := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
    a : ↑(F.obj A)
    h : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    b : ↑(F.obj A)
    ⊢ Eq (F.map u b) (F.map (CategoryTheory.PreGaloisCategory.selfProdPermIncl h b …
  -/
  apply Concrete.Pi.map_ext _ F
  /-
    case h
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
    a : ↑(F.obj A)
    h : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    b : ↑(F.obj A)
    ⊢ ∀ (i : ↑(F.obj X)), Eq ((F.map (CategoryTheory.Limits.Pi.π (fun x => X) i))  …
  -/
  intro (t : F.obj X)
  /-
    case h
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
    a : ↑(F.obj A)
    h : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    b : ↑(F.obj A)
    t : ↑(F.obj X)
    ⊢ Eq ((F.map (CategoryTheory.Limits.Pi.π (fun x => X) t)) (F.map u b)) ((F.map …
  -/
  convert_to F.map (selfProdProj u t) b = _
    /-
      case h.e'_2.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
      a : ↑(F.obj A)
      h : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
      b : ↑(F.obj A)
      t : ↑(F.obj X)
      e_1✝ : Eq ((CategoryTheory.forget FintypeCat).obj (F.obj X)) ↑(F.obj X)
      ⊢ Eq ((F.map (CategoryTheory.Limits.Pi.π (fun x => X) t)) (F.map u b)) (F.map  …
    -/
  · simp only [selfProdProj, map_comp, FintypeCat.comp_apply]; rfl
                                                               /-
                                                                 🎉 no goals
                                                               -/
    /-
      case h.convert_2
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
      a : ↑(F.obj A)
      h : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
      b : ↑(F.obj A)
      t : ↑(F.obj X)
      ⊢ Eq (F.map (CategoryTheory.PreGaloisCategory.selfProdProj u t) b) ((F.map (Ca …
    -/
  · dsimp only [selfProdPermIncl, Pi.whiskerEquiv]
    /-
      case h.convert_2
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
      a : ↑(F.obj A)
      h : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
      b : ↑(F.obj A)
      t : ↑(F.obj X)
      ⊢ Eq (F.map (CategoryTheory.PreGaloisCategory.selfProdProj u t) b) ((F.map (Ca …
    -/
    rw [map_comp, FintypeCat.comp_apply, h]
    convert_to F.map (selfProdProj u t) b =
      (F.map (Pi.map' (fiberPerm h b) fun _ ↦ 𝟙 X) ≫
      F.map (Pi.π (fun _ ↦ X) t)) (mkSelfProdFib F X)
    /-
      case h.convert_2
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
      a : ↑(F.obj A)
      h : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
      b : ↑(F.obj A)
      t : ↑(F.obj X)
      ⊢ Eq (F.map (CategoryTheory.PreGaloisCategory.selfProdProj u t) b) (CategoryTh …
    -/
    rw [← map_comp, Pi.map'_comp_π, Category.comp_id, mkSelfProdFib_map_π F X (fiberPerm h b t)]
    /-
      case h.convert_2
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
      a : ↑(F.obj A)
      h : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
      b : ↑(F.obj A)
      t : ↑(F.obj X)
      ⊢ Eq (F.map (CategoryTheory.PreGaloisCategory.selfProdProj u t) b) ((CategoryT …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- There exists an automorphism `f` of `A` that maps `b` to `a`.
`f` is obtained by considering `u` and `selfProdPermIncl h b`.
Both are inclusions of `A` into `selfProd F X` mapping `b` respectively `a` to the same element
in the fiber of `selfProd F X`. Applying `connected_component_unique` yields the result. -/
private lemma subobj_selfProd_trans [Mono u] (b : F.obj A) : ∃ (f : A ≅ A), F.map f.hom b = a := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{u₂, u₁} C
    inst✝³ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
    a : ↑(F.obj A)
    h : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝ : CategoryTheory.Mono u
    b : ↑(F.obj A)
    ⊢ Exists fun f => Eq (F.map f.hom b) a
  -/
  apply connected_component_unique F b a u (selfProdPermIncl h b)
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{u₂, u₁} C
    inst✝³ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
    a : ↑(F.obj A)
    h : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsConnected A
    inst✝ : CategoryTheory.Mono u
    b : ↑(F.obj A)
    ⊢ Eq (F.map u b) (F.map (CategoryTheory.PreGaloisCategory.selfProdPermIncl h b …
  -/
  exact selfProdTermIncl_fib_eq h b
  /-
    🎉 no goals
  -/


/-- The fiber of any object in a Galois category is represented by a Galois object. -/
lemma exists_galois_representative (X : C) : ∃ (A : C) (a : F.obj A),
    IsGalois A ∧ Function.Bijective (fun (f : A ⟶ X) ↦ F.map f a) := by
  obtain ⟨A, u, a, h1, h2, h3⟩ := fiber_in_connected_component F (selfProd F X)
    (mkSelfProdFib F X)
  /-
    case intro.intro.intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
    a : ↑(F.obj A)
    h1 : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
    h2 : CategoryTheory.PreGaloisCategory.IsConnected A
    h3 : CategoryTheory.Mono u
    ⊢ Exists fun A => Exists fun a => And (CategoryTheory.PreGaloisCategory.IsGalo …
  -/
  use A
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
    a : ↑(F.obj A)
    h1 : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
    h2 : CategoryTheory.PreGaloisCategory.IsConnected A
    h3 : CategoryTheory.Mono u
    ⊢ Exists fun a => And (CategoryTheory.PreGaloisCategory.IsGalois A) (Function. …
  -/
  use a
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
    a : ↑(F.obj A)
    h1 : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
    h2 : CategoryTheory.PreGaloisCategory.IsConnected A
    h3 : CategoryTheory.Mono u
    ⊢ And (CategoryTheory.PreGaloisCategory.IsGalois A) (Function.Bijective fun f  …
  -/
  constructor
    /-
      case h.left
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
      a : ↑(F.obj A)
      h1 : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
      h2 : CategoryTheory.PreGaloisCategory.IsConnected A
      h3 : CategoryTheory.Mono u
      ⊢ CategoryTheory.PreGaloisCategory.IsGalois A
    -/
  · refine (isGalois_iff_pretransitive F A).mpr ⟨fun x y ↦ ?_⟩
    /-
      case h.left
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
      a : ↑(F.obj A)
      h1 : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
      h2 : CategoryTheory.PreGaloisCategory.IsConnected A
      h3 : CategoryTheory.Mono u
      x y : ↑(F.obj A)
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    obtain ⟨fi1, hfi1⟩ := subobj_selfProd_trans h1 x
    /-
      case h.left.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
      a : ↑(F.obj A)
      h1 : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
      h2 : CategoryTheory.PreGaloisCategory.IsConnected A
      h3 : CategoryTheory.Mono u
      x y : ↑(F.obj A)
      fi1 : CategoryTheory.Iso A A
      hfi1 : Eq (F.map fi1.hom x) a
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    obtain ⟨fi2, hfi2⟩ := subobj_selfProd_trans h1 y
    /-
      case h.left.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
      a : ↑(F.obj A)
      h1 : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
      h2 : CategoryTheory.PreGaloisCategory.IsConnected A
      h3 : CategoryTheory.Mono u
      x y : ↑(F.obj A)
      fi1 : CategoryTheory.Iso A A
      hfi1 : Eq (F.map fi1.hom x) a
      fi2 : CategoryTheory.Iso A A
      hfi2 : Eq (F.map fi2.hom y) a
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    use fi1 ≪≫ fi2.symm
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
      a : ↑(F.obj A)
      h1 : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
      h2 : CategoryTheory.PreGaloisCategory.IsConnected A
      h3 : CategoryTheory.Mono u
      x y : ↑(F.obj A)
      fi1 : CategoryTheory.Iso A A
      hfi1 : Eq (F.map fi1.hom x) a
      fi2 : CategoryTheory.Iso A A
      hfi2 : Eq (F.map fi2.hom y) a
      ⊢ Eq (HSMul.hSMul (fi1.trans fi2.symm) x) y
    -/
    show F.map (fi1.hom ≫ fi2.inv) x = y
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
      a : ↑(F.obj A)
      h1 : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
      h2 : CategoryTheory.PreGaloisCategory.IsConnected A
      h3 : CategoryTheory.Mono u
      x y : ↑(F.obj A)
      fi1 : CategoryTheory.Iso A A
      hfi1 : Eq (F.map fi1.hom x) a
      fi2 : CategoryTheory.Iso A A
      hfi2 : Eq (F.map fi2.hom y) a
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp fi1.hom fi2.inv) x) y
    -/
    simp only [map_comp, FintypeCat.comp_apply]
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
      a : ↑(F.obj A)
      h1 : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
      h2 : CategoryTheory.PreGaloisCategory.IsConnected A
      h3 : CategoryTheory.Mono u
      x y : ↑(F.obj A)
      fi1 : CategoryTheory.Iso A A
      hfi1 : Eq (F.map fi1.hom x) a
      fi2 : CategoryTheory.Iso A A
      hfi2 : Eq (F.map fi2.hom y) a
      ⊢ Eq (F.map fi2.inv (F.map fi1.hom x)) y
    -/
    rw [hfi1, ← hfi2]
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
      a : ↑(F.obj A)
      h1 : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
      h2 : CategoryTheory.PreGaloisCategory.IsConnected A
      h3 : CategoryTheory.Mono u
      x y : ↑(F.obj A)
      fi1 : CategoryTheory.Iso A A
      hfi1 : Eq (F.map fi1.hom x) a
      fi2 : CategoryTheory.Iso A A
      hfi2 : Eq (F.map fi2.hom y) a
      ⊢ Eq (F.map fi2.inv (F.map fi2.hom y)) y
    -/
    exact congr_fun (F.mapIso fi2).hom_inv_id y
    /-
      🎉 no goals
    -/
    /-
      case h.right
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
      a : ↑(F.obj A)
      h1 : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
      h2 : CategoryTheory.PreGaloisCategory.IsConnected A
      h3 : CategoryTheory.Mono u
      ⊢ Function.Bijective fun f => F.map f a
    -/
  · refine ⟨evaluation_injective_of_isConnected F A X a, ?_⟩
    /-
      case h.right
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
      a : ↑(F.obj A)
      h1 : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
      h2 : CategoryTheory.PreGaloisCategory.IsConnected A
      h3 : CategoryTheory.Mono u
      ⊢ Function.Surjective fun f => F.map f a
    -/
    intro x
    /-
      case h.right
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
      a : ↑(F.obj A)
      h1 : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
      h2 : CategoryTheory.PreGaloisCategory.IsConnected A
      h3 : CategoryTheory.Mono u
      x : ↑(F.obj X)
      ⊢ Exists fun a_1 => Eq ((fun f => F.map f a) a_1) x
    -/
    use u ≫ Pi.π _ x
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      u : Quiver.Hom A (CategoryTheory.PreGaloisCategory.selfProd F X)
      a : ↑(F.obj A)
      h1 : Eq (F.map u a) (CategoryTheory.PreGaloisCategory.mkSelfProdFib F X)
      h2 : CategoryTheory.PreGaloisCategory.IsConnected A
      h3 : CategoryTheory.Mono u
      x : ↑(F.obj X)
      ⊢ Eq ((fun f => F.map f a) (CategoryTheory.CategoryStruct.comp u (CategoryTheo …
    -/
    exact (selfProdProj_fiber h1) x
    /-
      🎉 no goals
    -/


/-- Any element in the fiber of an object `X` is the evaluation of a morphism from a
Galois object. -/
lemma exists_hom_from_galois_of_fiber (X : C) (x : F.obj X) :
    ∃ (A : C) (f : A ⟶ X) (a : F.obj A), IsGalois A ∧ F.map f a = x := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    x : ↑(F.obj X)
    ⊢ Exists fun A => Exists fun f => Exists fun a => And (CategoryTheory.PreGaloi …
  -/
  obtain ⟨A, a, h1, h2⟩ := exists_galois_representative F X
  /-
    case intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    x : ↑(F.obj X)
    A : C
    a : ↑(F.obj A)
    h1 : CategoryTheory.PreGaloisCategory.IsGalois A
    h2 : Function.Bijective fun f => F.map f a
    ⊢ Exists fun A => Exists fun f => Exists fun a => And (CategoryTheory.PreGaloi …
  -/
  obtain ⟨f, hf⟩ := h2.surjective x
  /-
    case intro.intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    x : ↑(F.obj X)
    A : C
    a : ↑(F.obj A)
    h1 : CategoryTheory.PreGaloisCategory.IsGalois A
    h2 : Function.Bijective fun f => F.map f a
    f : Quiver.Hom A X
    hf : Eq ((fun f => F.map f a) f) x
    ⊢ Exists fun A => Exists fun f => Exists fun a => And (CategoryTheory.PreGaloi …
  -/
  exact ⟨A, f, a, h1, hf⟩
  /-
    🎉 no goals
  -/


/-- Any object with non-empty fiber admits a hom from a Galois object. -/
lemma exists_hom_from_galois_of_fiber_nonempty (X : C) (h : Nonempty (F.obj X)) :
    ∃ (A : C) (_ : A ⟶ X), IsGalois A := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    h : Nonempty ↑(F.obj X)
    ⊢ Exists fun A => Exists fun x => CategoryTheory.PreGaloisCategory.IsGalois A
  -/
  obtain ⟨x⟩ := h
  /-
    case intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    x : ↑(F.obj X)
    ⊢ Exists fun A => Exists fun x => CategoryTheory.PreGaloisCategory.IsGalois A
  -/
  obtain ⟨A, f, a, h1, _⟩ := exists_hom_from_galois_of_fiber F X x
  /-
    case intro.intro.intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    x : ↑(F.obj X)
    A : C
    f : Quiver.Hom A X
    a : ↑(F.obj A)
    h1 : CategoryTheory.PreGaloisCategory.IsGalois A
    right✝ : Eq (F.map f a) x
    ⊢ Exists fun A => Exists fun x => CategoryTheory.PreGaloisCategory.IsGalois A
  -/
  exact ⟨A, f, h1⟩
  /-
    🎉 no goals
  -/


include F in
/-- Any connected object admits a hom from a Galois object. -/
lemma exists_hom_from_galois_of_connected (X : C) [IsConnected X] :
    ∃ (A : C) (_ : A ⟶ X), IsGalois A :=
  exists_hom_from_galois_of_fiber_nonempty F X inferInstance


/-- To check equality of natural transformations `F ⟶ G`, it suffices to check it on
Galois objects. -/
lemma natTrans_ext_of_isGalois {G : C ⥤ FintypeCat.{w}} {t s : F ⟶ G}
    (h : ∀ (X : C) [IsGalois X], t.app X = s.app X) :
    t = s := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    G : CategoryTheory.Functor C FintypeCat
    t s : Quiver.Hom F G
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], Eq (t.app  …
    ⊢ Eq t s
  -/
  ext X x
  /-
    case w.h.h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    G : CategoryTheory.Functor C FintypeCat
    t s : Quiver.Hom F G
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], Eq (t.app  …
    X : C
    x : ↑(F.obj X)
    ⊢ Eq (t.app X x) (s.app X x)
  -/
  obtain ⟨A, f, a, _, rfl⟩ := exists_hom_from_galois_of_fiber F X x
  /-
    case w.h.h.intro.intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    G : CategoryTheory.Functor C FintypeCat
    t s : Quiver.Hom F G
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], Eq (t.app  …
    X A : C
    f : Quiver.Hom A X
    a : ↑(F.obj A)
    left✝ : CategoryTheory.PreGaloisCategory.IsGalois A
    ⊢ Eq (t.app X (F.map f a)) (s.app X (F.map f a))
  -/
  rw [FunctorToFintypeCat.naturality, FunctorToFintypeCat.naturality, h A]
  /-
    🎉 no goals
  -/


