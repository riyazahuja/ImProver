/--
The character module of an abelian group `A` in the unit rational circle is `A⋆ := Hom_ℤ(A, ℚ ⧸ ℤ)`.
-/
def CharacterModule : Type uA := A →+ AddCircle (1 : ℚ)


instance : FunLike (CharacterModule A) A (AddCircle (1 : ℚ)) where
  coe c := c.toFun
                             /-
                               R : Type uR
                               inst✝³ : CommRing R
                               A : Type uA
                               inst✝² : AddCommGroup A
                               A' : Type u_1
                               inst✝¹ : AddCommGroup A'
                               B : Type uB
                               inst✝ : AddCommGroup B
                               x✝² x✝¹ : CharacterModule A
                               x✝ : Eq ((fun c => (↑c).toFun) x✝²) ((fun c => (↑c).toFun) x✝¹)
                               ⊢ Eq x✝² x✝¹
                             -/
  coe_injective' _ _ _ := by aesop
                             /-
                               🎉 no goals
                             -/


instance : LinearMapClass (CharacterModule A) ℤ A (AddCircle (1 : ℚ)) where
                      /-
                        R : Type uR
                        inst✝³ : CommRing R
                        A : Type uA
                        inst✝² : AddCommGroup A
                        A' : Type u_1
                        inst✝¹ : AddCommGroup A'
                        B : Type uB
                        inst✝ : AddCommGroup B
                        x✝² : CharacterModule A
                        x✝¹ x✝ : A
                        ⊢ Eq (x✝² (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd (x✝² x✝¹) (x✝² x✝))
                      -/
  map_add _ _ _ := by rw [AddMonoidHom.map_add]
                      /-
                        🎉 no goals
                      -/
                         /-
                           R : Type uR
                           inst✝³ : CommRing R
                           A : Type uA
                           inst✝² : AddCommGroup A
                           A' : Type u_1
                           inst✝¹ : AddCommGroup A'
                           B : Type uB
                           inst✝ : AddCommGroup B
                           x✝² : CharacterModule A
                           x✝¹ : Int
                           x✝ : A
                           ⊢ Eq (x✝² (HSMul.hSMul x✝¹ x✝)) (HSMul.hSMul ((RingHom.id Int) x✝¹) (x✝² x✝))
                         -/
  map_smulₛₗ _ _ _ := by rw [AddMonoidHom.map_zsmul, RingHom.id_apply]
                         /-
                           🎉 no goals
                         -/


instance : AddCommGroup (CharacterModule A) :=
  inferInstanceAs (AddCommGroup (A →+ _))


@[ext] theorem ext {c c' : CharacterModule A} (h : ∀ x, c x = c' x) : c = c' := DFunLike.ext _ _ h


instance : Module R (CharacterModule A) :=
  Module.compHom (A →+ _) (RingEquiv.toOpposite _ |>.toRingHom : R →+* Rᵈᵐᵃ)


@[simp] lemma smul_apply (c : CharacterModule A) (r : R) (a : A) : (r • c) a = c (r • a) := rfl


/--
Given an abelian group homomorphism `f : A → B`, `f⋆(L) := L ∘ f` defines a linear map
from `B⋆` to `A⋆`.
-/
@[simps] def dual (f : A →ₗ[R] B) : CharacterModule B →ₗ[R] CharacterModule A where
  toFun L := L.comp f.toAddMonoidHom
                 /-
                   R : Type uR
                   inst✝⁶ : CommRing R
                   A : Type uA
                   inst✝⁵ : AddCommGroup A
                   A' : Type u_1
                   inst✝⁴ : AddCommGroup A'
                   B : Type uB
                   inst✝³ : AddCommGroup B
                   inst✝² : Module R A
                   inst✝¹ : Module R A'
                   inst✝ : Module R B
                   f : LinearMap (RingHom.id R) A B
                   ⊢ ∀ (x y : CharacterModule B), Eq ((fun L => AddMonoidHom.comp L f.toAddMonoid …
                 -/
  map_add' := by aesop
                 /-
                   🎉 no goals
                 -/
                      /-
                        R : Type uR
                        inst✝⁶ : CommRing R
                        A : Type uA
                        inst✝⁵ : AddCommGroup A
                        A' : Type u_1
                        inst✝⁴ : AddCommGroup A'
                        B : Type uB
                        inst✝³ : AddCommGroup B
                        inst✝² : Module R A
                        inst✝¹ : Module R A'
                        inst✝ : Module R B
                        f : LinearMap (RingHom.id R) A B
                        r : R
                        c : CharacterModule B
                        ⊢ Eq ({ toFun := fun L => AddMonoidHom.comp L f.toAddMonoidHom, map_add' := ⋯  …
                      -/
  map_smul' r c := by ext x; exact congr(c $(f.map_smul r x)).symm
                             /-
                               🎉 no goals
                             -/


lemma dual_surjective_of_injective (f : A →ₗ[R] B) (hf : Function.Injective f) :
    Function.Surjective (dual f) :=
  (Module.Baer.of_divisible _).extension_property_addMonoidHom _ hf


/--
Two isomorphic modules have isomorphic character modules.
-/
def congr (e : A ≃ₗ[R] B) : CharacterModule A ≃ₗ[R] CharacterModule B :=
  .ofLinear (dual e.symm) (dual e)
        /-
          R : Type uR
          inst✝⁶ : CommRing R
          A : Type uA
          inst✝⁵ : AddCommGroup A
          A' : Type u_1
          inst✝⁴ : AddCommGroup A'
          B : Type uB
          inst✝³ : AddCommGroup B
          inst✝² : Module R A
          inst✝¹ : Module R A'
          inst✝ : Module R B
          e : LinearEquiv (RingHom.id R) A B
          ⊢ Eq ((CharacterModule.dual ↑e.symm).comp (CharacterModule.dual ↑e)) LinearMap …
        -/
    (by ext c _; exact congr(c $(e.right_inv _)))
                 /-
                   🎉 no goals
                 -/
        /-
          R : Type uR
          inst✝⁶ : CommRing R
          A : Type uA
          inst✝⁵ : AddCommGroup A
          A' : Type u_1
          inst✝⁴ : AddCommGroup A'
          B : Type uB
          inst✝³ : AddCommGroup B
          inst✝² : Module R A
          inst✝¹ : Module R A'
          inst✝ : Module R B
          e : LinearEquiv (RingHom.id R) A B
          ⊢ Eq ((CharacterModule.dual ↑e).comp (CharacterModule.dual ↑e.symm)) LinearMap …
        -/
    (by ext c _; exact congr(c $(e.left_inv _)))
                 /-
                   🎉 no goals
                 -/


/--
Any linear map `L : A → B⋆` induces a character in `(A ⊗ B)⋆` by `a ⊗ b ↦ L a b`.
-/
@[simps] noncomputable def uncurry :
    (A →ₗ[R] CharacterModule B) →ₗ[R] CharacterModule (A ⊗[R] B) where
  toFun c := TensorProduct.liftAddHom c.toAddMonoidHom fun r a b ↦ congr($(c.map_smul r a) b)
                                               /-
                                                 R : Type uR
                                                 inst✝⁶ : CommRing R
                                                 A : Type uA
                                                 inst✝⁵ : AddCommGroup A
                                                 A' : Type u_1
                                                 inst✝⁴ : AddCommGroup A'
                                                 B : Type uB
                                                 inst✝³ : AddCommGroup B
                                                 inst✝² : Module R A
                                                 inst✝¹ : Module R A'
                                                 inst✝ : Module R B
                                                 c c' : LinearMap (RingHom.id R) A (CharacterModule B)
                                                 x : TensorProduct R A B
                                                 ⊢ Eq (((fun c => TensorProduct.liftAddHom c.toAddMonoidHom ⋯) (HAdd.hAdd c c') …
                                               -/
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
  map_add' c c' := DFunLike.ext _ _ fun x ↦ by refine x.induction_on ?_ ?_ ?_ <;> aesop
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
  map_smul' r c := DFunLike.ext _ _ fun x ↦ x.induction_on
        /-
          R : Type uR
          inst✝⁶ : CommRing R
          A : Type uA
          inst✝⁵ : AddCommGroup A
          A' : Type u_1
          inst✝⁴ : AddCommGroup A'
          B : Type uB
          inst✝³ : AddCommGroup B
          inst✝² : Module R A
          inst✝¹ : Module R A'
          inst✝ : Module R B
          r : R
          c : LinearMap (RingHom.id R) A (CharacterModule B)
          x : TensorProduct R A B
          ⊢ Eq (({ toFun := fun c => TensorProduct.liftAddHom c.toAddMonoidHom ⋯, map_ad …
        -/
        /-
          🎉 no goals
        -/
    (by simp_rw [map_zero]) (fun a b ↦ congr($(c.map_smul r a) b).symm) (by aesop)
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/--
Any character `c` in `(A ⊗ B)⋆` induces a linear map `A → B⋆` by `a ↦ b ↦ c (a ⊗ b)`.
-/
@[simps] noncomputable def curry :
    CharacterModule (A ⊗[R] B) →ₗ[R] (A →ₗ[R] CharacterModule B) where
  toFun c :=
  { toFun := (c.comp <| TensorProduct.mk R A B ·)
    map_add' := fun _ _ ↦ DFunLike.ext _ _ fun b ↦
      congr(c <| $(map_add (mk R A B) _ _) b).trans (c.map_add _ _)
                              /-
                                R : Type uR
                                inst✝⁶ : CommRing R
                                A : Type uA
                                inst✝⁵ : AddCommGroup A
                                A' : Type u_1
                                inst✝⁴ : AddCommGroup A'
                                B : Type uB
                                inst✝³ : AddCommGroup B
                                inst✝² : Module R A
                                inst✝¹ : Module R A'
                                inst✝ : Module R B
                                c : CharacterModule (TensorProduct R A B)
                                r : R
                                a : A
                                ⊢ Eq ({ toFun := fun x => AddMonoidHom.comp c ↑((TensorProduct.mk R A B) x), m …
                              -/
    map_smul' := fun r a ↦ by ext; exact congr(c $(TensorProduct.tmul_smul _ _ _)).symm }
                                   /-
                                     🎉 no goals
                                   -/
  map_add' _ _ := rfl
                      /-
                        R : Type uR
                        inst✝⁶ : CommRing R
                        A : Type uA
                        inst✝⁵ : AddCommGroup A
                        A' : Type u_1
                        inst✝⁴ : AddCommGroup A'
                        B : Type uB
                        inst✝³ : AddCommGroup B
                        inst✝² : Module R A
                        inst✝¹ : Module R A'
                        inst✝ : Module R B
                        r : R
                        c : CharacterModule (TensorProduct R A B)
                        ⊢ Eq ({ toFun := fun c => { toFun := fun x => AddMonoidHom.comp c ↑((TensorPro …
                      -/
  map_smul' r c := by ext; exact congr(c $(TensorProduct.tmul_smul _ _ _)).symm
                           /-
                             🎉 no goals
                           -/


/--
Linear maps into a character module are exactly characters of the tensor product.
-/
@[simps!] noncomputable def homEquiv :
    (A →ₗ[R] CharacterModule B) ≃ₗ[R] CharacterModule (A ⊗[R] B) :=
                              /-
                                R : Type uR
                                inst✝⁶ : CommRing R
                                A : Type uA
                                inst✝⁵ : AddCommGroup A
                                A' : Type u_1
                                inst✝⁴ : AddCommGroup A'
                                B : Type uB
                                inst✝³ : AddCommGroup B
                                inst✝² : Module R A
                                inst✝¹ : Module R A'
                                inst✝ : Module R B
                                ⊢ Eq (CharacterModule.uncurry.comp CharacterModule.curry) LinearMap.id
                              -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
  .ofLinear uncurry curry (by ext _ z; refine z.induction_on ?_ ?_ ?_ <;> aesop) (by aesop)
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


theorem dual_rTensor_conj_homEquiv (f : A →ₗ[R] A') :
    homEquiv.symm.toLinearMap ∘ₗ dual (f.rTensor B) ∘ₗ homEquiv.toLinearMap = f.lcomp R _ := rfl


/--
`ℤ⋆`, the character module of `ℤ` in the unit rational circle.
-/
protected abbrev int : Type := CharacterModule ℤ


/-- Given `n : ℕ`, the map `m ↦ m / n`. -/
protected abbrev int.divByNat (n : ℕ) : CharacterModule.int :=
  LinearMap.toSpanSingleton ℤ _ (QuotientAddGroup.mk (n : ℚ)⁻¹) |>.toAddMonoidHom


protected lemma int.divByNat_self (n : ℕ) :
    int.divByNat n n = 0 := by
  /-
    n : Nat
    ⊢ Eq ((CharacterModule.int.divByNat n) ↑n) 0
  -/
  obtain rfl | h0 := eq_or_ne n 0
    /-
      case inl
      ⊢ Eq ((CharacterModule.int.divByNat 0) ↑0) 0
    -/
  · apply map_zero
    /-
      🎉 no goals
    -/
  exact (AddCircle.coe_eq_zero_iff _).mpr
    ⟨1, by simp [mul_inv_cancel₀ (Nat.cast_ne_zero (R := ℚ).mpr h0)]⟩


/-- The `ℤ`-submodule spanned by a single element `a` is isomorphic to the quotient of `ℤ`
by the ideal generated by the order of `a`. -/
@[simps!] noncomputable def intSpanEquivQuotAddOrderOf (a : A) :
    (ℤ ∙ a) ≃ₗ[ℤ] ℤ ⧸ Ideal.span {(addOrderOf a : ℤ)} :=
  LinearEquiv.ofEq _ _ (LinearMap.span_singleton_eq_range ℤ A a) ≪≫ₗ
  (LinearMap.quotKerEquivRange <| LinearMap.toSpanSingleton ℤ A a).symm ≪≫ₗ
  Submodule.quotEquivOfEq _ _ (by
    /-
      R : Type uR
      inst✝³ : CommRing R
      A : Type uA
      inst✝² : AddCommGroup A
      A' : Type u_1
      inst✝¹ : AddCommGroup A'
      B : Type uB
      inst✝ : AddCommGroup B
      a : A
      ⊢ Eq (LinearMap.ker (LinearMap.toSpanSingleton Int A a)) (Ideal.span (Singleto …
    -/
    ext1 x
    rw [Ideal.mem_span_singleton, addOrderOf_dvd_iff_zsmul_eq_zero, LinearMap.mem_ker,
      LinearMap.toSpanSingleton_apply])


lemma intSpanEquivQuotAddOrderOf_apply_self (a : A) :
    intSpanEquivQuotAddOrderOf a ⟨a, Submodule.mem_span_singleton_self a⟩ =
    Submodule.Quotient.mk 1 :=
  (LinearEquiv.eq_symm_apply _).mp <| Subtype.ext (one_zsmul _).symm


/--
For an abelian group `A` and an element `a ∈ A`, there is a character `c : ℤ ∙ a → ℚ ⧸ ℤ` given by
`m • a ↦ m / n` where `n` is the smallest positive integer such that `n • a = 0` and when such `n`
does not exist, `c` is defined by `m • a ↦ m / 2`.
-/
noncomputable def ofSpanSingleton (a : A) : CharacterModule (ℤ ∙ a) :=
  let l :  ℤ ⧸ Ideal.span {(addOrderOf a : ℤ)} →ₗ[ℤ] AddCircle (1 : ℚ) :=
    Submodule.liftQSpanSingleton _
      (CharacterModule.int.divByNat <|
        if addOrderOf a = 0 then 2 else addOrderOf a).toIntLinearMap <| by
        /-
          R : Type uR
          inst✝³ : CommRing R
          A : Type uA
          inst✝² : AddCommGroup A
          A' : Type u_1
          inst✝¹ : AddCommGroup A'
          B : Type uB
          inst✝ : AddCommGroup B
          a : A
          ⊢ Eq ((AddMonoidHom.toIntLinearMap (CharacterModule.int.divByNat (ite (Eq (add …
        -/
        split_ifs with h
          /-
            case pos
            R : Type uR
            inst✝³ : CommRing R
            A : Type uA
            inst✝² : AddCommGroup A
            A' : Type u_1
            inst✝¹ : AddCommGroup A'
            B : Type uB
            inst✝ : AddCommGroup B
            a : A
            h : Eq (addOrderOf a) 0
            ⊢ Eq ((AddMonoidHom.toIntLinearMap (CharacterModule.int.divByNat 2)) ↑(addOrde …
          -/
        · rw [h, Nat.cast_zero, map_zero]
          /-
            🎉 no goals
          -/
          /-
            case neg
            R : Type uR
            inst✝³ : CommRing R
            A : Type uA
            inst✝² : AddCommGroup A
            A' : Type u_1
            inst✝¹ : AddCommGroup A'
            B : Type uB
            inst✝ : AddCommGroup B
            a : A
            h : Not (Eq (addOrderOf a) 0)
            ⊢ Eq ((AddMonoidHom.toIntLinearMap (CharacterModule.int.divByNat (addOrderOf a …
          -/
        · apply CharacterModule.int.divByNat_self
          /-
            🎉 no goals
          -/
  l ∘ₗ intSpanEquivQuotAddOrderOf a |>.toAddMonoidHom


lemma eq_zero_of_ofSpanSingleton_apply_self (a : A)
    (h : ofSpanSingleton a ⟨a, Submodule.mem_span_singleton_self a⟩ = 0) : a = 0 := by
  erw [ofSpanSingleton, LinearMap.toAddMonoidHom_coe, LinearMap.comp_apply,
     intSpanEquivQuotAddOrderOf_apply_self, Submodule.liftQSpanSingleton_apply,
    AddMonoidHom.coe_toIntLinearMap, int.divByNat, LinearMap.toSpanSingleton_one,
    AddCircle.coe_eq_zero_iff] at h
  /-
    A : Type uA
    inst✝ : AddCommGroup A
    a : A
    h : Exists fun n => Eq (HSMul.hSMul n 1) (Inv.inv ↑(ite (Eq (addOrderOf a) 0)  …
    ⊢ Eq a 0
  -/
  rcases h with ⟨n, hn⟩
  /-
    case intro
    A : Type uA
    inst✝ : AddCommGroup A
    a : A
    n : Int
    hn : Eq (HSMul.hSMul n 1) (Inv.inv ↑(ite (Eq (addOrderOf a) 0) 2 (addOrderOf a …
    ⊢ Eq a 0
  -/
  apply_fun Rat.den at hn
  /-
    case intro
    A : Type uA
    inst✝ : AddCommGroup A
    a : A
    n : Int
    hn : Eq (HSMul.hSMul n 1).den (Inv.inv ↑(ite (Eq (addOrderOf a) 0) 2 (addOrder …
    ⊢ Eq a 0
  -/
  rw [zsmul_one, Rat.den_intCast, Rat.inv_natCast_den_of_pos] at hn
    /-
      case intro
      A : Type uA
      inst✝ : AddCommGroup A
      a : A
      n : Int
      hn : Eq 1 (ite (Eq (addOrderOf a) 0) 2 (addOrderOf a))
      ⊢ Eq a 0
    -/
  · split_ifs at hn
      /-
        case pos
        A : Type uA
        inst✝ : AddCommGroup A
        a : A
        n : Int
        h✝ : Eq (addOrderOf a) 0
        hn : Eq 1 2
        ⊢ Eq a 0
      -/
    · cases hn
      /-
        🎉 no goals
      -/
      /-
        case neg
        A : Type uA
        inst✝ : AddCommGroup A
        a : A
        n : Int
        h✝ : Not (Eq (addOrderOf a) 0)
        hn : Eq 1 (addOrderOf a)
        ⊢ Eq a 0
      -/
    · rwa [eq_comm, AddMonoid.addOrderOf_eq_one_iff] at hn
      /-
        🎉 no goals
      -/
    /-
      case intro
      A : Type uA
      inst✝ : AddCommGroup A
      a : A
      n : Int
      hn : Eq 1 (Inv.inv ↑(ite (Eq (addOrderOf a) 0) 2 (addOrderOf a))).den
      ⊢ LT.lt 0 (ite (Eq (addOrderOf a) 0) 2 (addOrderOf a))
    -/
  · split_ifs with h
      /-
        case pos
        A : Type uA
        inst✝ : AddCommGroup A
        a : A
        n : Int
        hn : Eq 1 (Inv.inv ↑(ite (Eq (addOrderOf a) 0) 2 (addOrderOf a))).den
        h : Eq (addOrderOf a) 0
        ⊢ LT.lt 0 2
      -/
    · norm_num
      /-
        🎉 no goals
      -/
      /-
        case neg
        A : Type uA
        inst✝ : AddCommGroup A
        a : A
        n : Int
        hn : Eq 1 (Inv.inv ↑(ite (Eq (addOrderOf a) 0) 2 (addOrderOf a))).den
        h : Not (Eq (addOrderOf a) 0)
        ⊢ LT.lt 0 (addOrderOf a)
      -/
    · exact Nat.pos_of_ne_zero h
      /-
        🎉 no goals
      -/


lemma exists_character_apply_ne_zero_of_ne_zero {a : A} (ne_zero : a ≠ 0) :
    ∃ (c : CharacterModule A), c a ≠ 0 :=
  have ⟨c, hc⟩ := dual_surjective_of_injective _ (Submodule.injective_subtype _) (ofSpanSingleton a)
                                                                       /-
                                                                         A : Type uA
                                                                         inst✝ : AddCommGroup A
                                                                         a : A
                                                                         ne_zero : Ne a 0
                                                                         c : CharacterModule A
                                                                         hc : Eq ((CharacterModule.dual (Submodule.span Int (Singleton.singleton a)).su …
                                                                         h : Eq (c a) 0
                                                                         ⊢ Eq ((CharacterModule.ofSpanSingleton a) ⟨a, ⋯⟩) 0
                                                                       -/
  ⟨c, fun h ↦ ne_zero <| eq_zero_of_ofSpanSingleton_apply_self a <| by rwa [← hc]⟩
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


lemma eq_zero_of_character_apply {a : A} (h : ∀ c : CharacterModule A, c a = 0) : a = 0 := by
  /-
    A : Type uA
    inst✝ : AddCommGroup A
    a : A
    h : ∀ (c : CharacterModule A), Eq (c a) 0
    ⊢ Eq a 0
  -/
  contrapose! h; exact exists_character_apply_ne_zero_of_ne_zero h
                 /-
                   🎉 no goals
                 -/


lemma dual_surjective_iff_injective {f : A →ₗ[R] A'} :
    Function.Surjective (dual f) ↔ Function.Injective f :=
  ⟨fun h ↦ (injective_iff_map_eq_zero _).2 fun a h0 ↦ eq_zero_of_character_apply fun c ↦ by
    /-
      R : Type uR
      inst✝⁴ : CommRing R
      A : Type uA
      inst✝³ : AddCommGroup A
      A' : Type u_1
      inst✝² : AddCommGroup A'
      inst✝¹ : Module R A
      inst✝ : Module R A'
      f : LinearMap (RingHom.id R) A A'
      h : Function.Surjective ⇑(CharacterModule.dual f)
      a : A
      h0 : Eq (f a) 0
      c : CharacterModule A
      ⊢ Eq (c a) 0
    -/
    obtain ⟨c, rfl⟩ := h c; exact congr(c $h0).trans c.map_zero,
                            /-
                              🎉 no goals
                            -/
  dual_surjective_of_injective f⟩


theorem _root_.rTensor_injective_iff_lcomp_surjective {f : A →ₗ[R] A'} :
    Function.Injective (f.rTensor B) ↔ Function.Surjective (f.lcomp R <| CharacterModule B) := by
  /-
    R : Type uR
    inst✝⁶ : CommRing R
    A : Type uA
    inst✝⁵ : AddCommGroup A
    A' : Type u_1
    inst✝⁴ : AddCommGroup A'
    B : Type uB
    inst✝³ : AddCommGroup B
    inst✝² : Module R A
    inst✝¹ : Module R A'
    inst✝ : Module R B
    f : LinearMap (RingHom.id R) A A'
    ⊢ Iff (Function.Injective ⇑(LinearMap.rTensor B f)) (Function.Surjective ⇑(Lin …
  -/
  simp [← dual_rTensor_conj_homEquiv, dual_surjective_iff_injective]
  /-
    🎉 no goals
  -/


