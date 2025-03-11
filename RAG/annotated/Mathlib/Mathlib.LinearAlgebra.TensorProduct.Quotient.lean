/--
Let `M, N` be `R`-modules, `m ≤ M` and `n ≤ N` be an `R`-submodules. Then we have a linear
isomorphism between tensor products of the quotients and the quotient of the tensor product:
`(M ⧸ m) ⊗[R] (N ⧸ n) ≃ₗ[R] (M ⊗[R] N) ⧸ (m ⊗ N ⊔ M ⊗ n)`.
-/
noncomputable def quotientTensorQuotientEquiv (m : Submodule R M) (n : Submodule R N) :
    (M ⧸ (m : Submodule R M)) ⊗[R] (N ⧸ (n : Submodule R N)) ≃ₗ[R]
    (M ⊗[R] N) ⧸
      (LinearMap.range (map m.subtype LinearMap.id) ⊔
        LinearMap.range (map LinearMap.id n.subtype)) :=
  LinearEquiv.ofLinear
    (lift <| Submodule.liftQ _ (LinearMap.flip <| Submodule.liftQ _
      ((mk R (M := M) (N := N)).flip.compr₂ (Submodule.mkQ _)) fun x hx => by
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        m : Submodule R M
        n : Submodule R N
        x : N
        hx : Membership.mem n x
        ⊢ Membership.mem (LinearMap.ker ((TensorProduct.mk R M N).flip.compr₂ (Max.max …
      -/
      ext y
      simp only [LinearMap.compr₂_apply, LinearMap.flip_apply, mk_apply, Submodule.mkQ_apply,
        LinearMap.zero_apply, Submodule.Quotient.mk_eq_zero]
      /-
        case h
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        m : Submodule R M
        n : Submodule R N
        x : N
        hx : Membership.mem n x
        y : M
        ⊢ Membership.mem (Max.max (LinearMap.range (TensorProduct.map m.subtype Linear …
      -/
      exact Submodule.mem_sup_right ⟨y ⊗ₜ ⟨x, hx⟩, rfl⟩) fun x hx => by
      /-
        🎉 no goals
      -/
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        m : Submodule R M
        n : Submodule R N
        x : M
        hx : Membership.mem m x
        ⊢ Membership.mem (LinearMap.ker (n.liftQ ((TensorProduct.mk R M N).flip.compr₂ …
      -/
      ext y
      simp only [LinearMap.coe_comp, Function.comp_apply, Submodule.mkQ_apply, LinearMap.flip_apply,
        Submodule.liftQ_apply, LinearMap.compr₂_apply, mk_apply, LinearMap.zero_comp,
        LinearMap.zero_apply, Submodule.Quotient.mk_eq_zero]
      /-
        case h.h
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        m : Submodule R M
        n : Submodule R N
        x : M
        hx : Membership.mem m x
        y : N
        ⊢ Membership.mem (Max.max (LinearMap.range (TensorProduct.map m.subtype Linear …
      -/
      exact Submodule.mem_sup_left ⟨⟨x, hx⟩ ⊗ₜ y, rfl⟩)
      /-
        🎉 no goals
      -/
    (Submodule.liftQ _ (map (Submodule.mkQ _) (Submodule.mkQ _)) fun x hx => by
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        m : Submodule R M
        n : Submodule R N
        x : TensorProduct R M N
        hx : Membership.mem (Max.max (LinearMap.range (TensorProduct.map m.subtype Lin …
        ⊢ Membership.mem (LinearMap.ker (TensorProduct.map m.mkQ n.mkQ)) x
      -/
      rw [Submodule.mem_sup] at hx
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        m : Submodule R M
        n : Submodule R N
        x : TensorProduct R M N
        hx : Exists fun y => And (Membership.mem (LinearMap.range (TensorProduct.map m …
        ⊢ Membership.mem (LinearMap.ker (TensorProduct.map m.mkQ n.mkQ)) x
      -/
      rcases hx with ⟨_, ⟨a, rfl⟩, _, ⟨b, rfl⟩, rfl⟩
      /-
        case intro.intro.intro.intro.intro.intro
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        m : Submodule R M
        n : Submodule R N
        a : TensorProduct R (Subtype fun x => Membership.mem m x) N
        b : TensorProduct R M (Subtype fun x => Membership.mem n x)
        ⊢ Membership.mem (LinearMap.ker (TensorProduct.map m.mkQ n.mkQ)) (HAdd.hAdd (( …
      -/
      simp only [LinearMap.mem_ker, map_add]
      /-
        case intro.intro.intro.intro.intro.intro
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        m : Submodule R M
        n : Submodule R N
        a : TensorProduct R (Subtype fun x => Membership.mem m x) N
        b : TensorProduct R M (Subtype fun x => Membership.mem n x)
        ⊢ Eq (HAdd.hAdd ((TensorProduct.map m.mkQ n.mkQ) ((TensorProduct.map m.subtype …
      -/
      set f := (map m.mkQ n.mkQ) ∘ₗ (map m.subtype LinearMap.id)
      /-
        case intro.intro.intro.intro.intro.intro
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        m : Submodule R M
        n : Submodule R N
        a : TensorProduct R (Subtype fun x => Membership.mem m x) N
        b : TensorProduct R M (Subtype fun x => Membership.mem n x)
        f : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.mem …
        ⊢ Eq (HAdd.hAdd ((TensorProduct.map m.mkQ n.mkQ) ((TensorProduct.map m.subtype …
      -/
      set g := (map m.mkQ n.mkQ) ∘ₗ (map LinearMap.id n.subtype)
      have eq : LinearMap.coprod f g = 0 := by
        ext x y
        · simp [f, Submodule.Quotient.mk_eq_zero _ |>.2 x.2]
        · simp [g, Submodule.Quotient.mk_eq_zero _ |>.2 y.2]
      /-
        case intro.intro.intro.intro.intro.intro
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        m : Submodule R M
        n : Submodule R N
        a : TensorProduct R (Subtype fun x => Membership.mem m x) N
        b : TensorProduct R M (Subtype fun x => Membership.mem n x)
        f : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.mem …
        g : LinearMap (RingHom.id R) (TensorProduct R M (Subtype fun x => Membership.m …
        eq : Eq (f.coprod g) 0
        ⊢ Eq (HAdd.hAdd ((TensorProduct.map m.mkQ n.mkQ) ((TensorProduct.map m.subtype …
      -/
      exact congr($eq (a, b)))
      /-
        🎉 no goals
      -/
        /-
          R : Type u_1
          M : Type u_2
          N : Type u_3
          inst✝⁴ : CommRing R
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : AddCommGroup N
          inst✝ : Module R N
          m : Submodule R M
          n : Submodule R N
          ⊢ Eq ((TensorProduct.lift (m.liftQ (n.liftQ ((TensorProduct.mk R M N).flip.com …
        -/
             /-
               🎉 no goals
             -/
    (by ext; simp) (by ext; simp)
                            /-
                              🎉 no goals
                            -/


@[simp]
lemma quotientTensorQuotientEquiv_apply_tmul_mk_tmul_mk
    (m : Submodule R M) (n : Submodule R N) (x : M) (y : N) :
    quotientTensorQuotientEquiv m n
      (Submodule.Quotient.mk x ⊗ₜ[R] Submodule.Quotient.mk y) =
      Submodule.Quotient.mk (x ⊗ₜ y) := rfl


@[simp]
lemma quotientTensorQuotientEquiv_symm_apply_mk_tmul
    (m : Submodule R M) (n : Submodule R N) (x : M) (y : N) :
    (quotientTensorQuotientEquiv m n).symm (Submodule.Quotient.mk (x ⊗ₜ y)) =
      Submodule.Quotient.mk x ⊗ₜ[R] Submodule.Quotient.mk y := rfl


variable (N) in
/--
Let `M, N` be `R`-modules, `m ≤ M` be an `R`-submodule. Then we have a linear isomorphism between
tensor products of the quotient and the quotient of the tensor product:
`(M ⧸ m) ⊗[R] N ≃ₗ[R] (M ⊗[R] N) ⧸ (m ⊗ N)`.
-/
noncomputable def quotientTensorEquiv (m : Submodule R M) :
    (M ⧸ (m : Submodule R M)) ⊗[R] N ≃ₗ[R]
    (M ⊗[R] N) ⧸ (LinearMap.range (map m.subtype (LinearMap.id : N →ₗ[R] N))) :=
  congr (LinearEquiv.refl _ _) ((Submodule.quotEquivOfEqBot _ rfl).symm) ≪≫ₗ
  quotientTensorQuotientEquiv (N := N) m ⊥ ≪≫ₗ
  Submodule.Quotient.equiv _ _ (LinearEquiv.refl _ _) (by
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      m : Submodule R M
      ⊢ Eq (Submodule.map (LinearEquiv.refl R (TensorProduct R M N)) (Max.max (Linea …
    -/
    simp only [Submodule.map_sup]
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      m : Submodule R M
      ⊢ Eq (Max.max (Submodule.map (LinearEquiv.refl R (TensorProduct R M N)) (Linea …
    -/
    erw [Submodule.map_id, Submodule.map_id]
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      m : Submodule R M
      ⊢ Eq (Max.max (LinearMap.range (TensorProduct.map m.subtype LinearMap.id)) (Li …
    -/
    simp only [sup_eq_left]
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      m : Submodule R M
      ⊢ LE.le (LinearMap.range (TensorProduct.map LinearMap.id Bot.bot.subtype)) (Li …
    -/
    rw [map_range_eq_span_tmul, map_range_eq_span_tmul]
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      m : Submodule R M
      ⊢ LE.le (Submodule.span R (setOf fun t => Exists fun m => Exists fun n => Eq ( …
    -/
    aesop)
    /-
      🎉 no goals
    -/


@[simp]
lemma quotientTensorEquiv_apply_tmul_mk (m : Submodule R M) (x : M) (y : N) :
    quotientTensorEquiv N m (Submodule.Quotient.mk x ⊗ₜ[R] y) =
    Submodule.Quotient.mk (x ⊗ₜ y) :=
  rfl


@[simp]
lemma quotientTensorEquiv_symm_apply_mk_tmul (m : Submodule R M) (x : M) (y : N) :
    (quotientTensorEquiv N m).symm (Submodule.Quotient.mk (x ⊗ₜ y)) =
    Submodule.Quotient.mk x ⊗ₜ[R] y :=
  rfl


variable (M) in
/--
Let `M, N` be `R`-modules, `n ≤ N` be an `R`-submodule. Then we have a linear isomorphism between
tensor products of the quotient and the quotient of the tensor product:
`M ⊗[R] (N ⧸ n) ≃ₗ[R] (M ⊗[R] N) ⧸ (M ⊗ n)`.
-/
noncomputable def tensorQuotientEquiv (n : Submodule R N) :
    M ⊗[R] (N ⧸ (n : Submodule R N)) ≃ₗ[R]
    (M ⊗[R] N) ⧸ (LinearMap.range (map (LinearMap.id : M →ₗ[R] M) n.subtype)) :=
  congr ((Submodule.quotEquivOfEqBot _ rfl).symm) (LinearEquiv.refl _ _)  ≪≫ₗ
  quotientTensorQuotientEquiv (⊥ : Submodule R M) n ≪≫ₗ
  Submodule.Quotient.equiv _ _ (LinearEquiv.refl _ _) (by
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      n : Submodule R N
      ⊢ Eq (Submodule.map (LinearEquiv.refl R (TensorProduct R M N)) (Max.max (Linea …
    -/
    simp only [Submodule.map_sup]
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      n : Submodule R N
      ⊢ Eq (Max.max (Submodule.map (LinearEquiv.refl R (TensorProduct R M N)) (Linea …
    -/
    erw [Submodule.map_id, Submodule.map_id]
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      n : Submodule R N
      ⊢ Eq (Max.max (LinearMap.range (TensorProduct.map Bot.bot.subtype LinearMap.id …
    -/
    simp only [sup_eq_right]
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      n : Submodule R N
      ⊢ LE.le (LinearMap.range (TensorProduct.map Bot.bot.subtype LinearMap.id)) (Li …
    -/
    rw [map_range_eq_span_tmul, map_range_eq_span_tmul]
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      n : Submodule R N
      ⊢ LE.le (Submodule.span R (setOf fun t => Exists fun m => Exists fun n => Eq ( …
    -/
    aesop)
    /-
      🎉 no goals
    -/


@[simp]
lemma tensorQuotientEquiv_apply_mk_tmul (n : Submodule R N) (x : M) (y : N) :
    tensorQuotientEquiv M n (x ⊗ₜ[R] Submodule.Quotient.mk y) =
    Submodule.Quotient.mk (x ⊗ₜ y) :=
  rfl


@[simp]
lemma tensorQuotientEquiv_symm_apply_tmul_mk (n : Submodule R N) (x : M) (y : N) :
    (tensorQuotientEquiv M n).symm (Submodule.Quotient.mk (x ⊗ₜ y)) =
    x ⊗ₜ[R] Submodule.Quotient.mk y :=
  rfl


variable (M) in
/-- Left tensoring a module with a quotient of the ring is the same as
quotienting that module by the corresponding submodule. -/
noncomputable def quotTensorEquivQuotSMul (I : Ideal R) :
    ((R ⧸ I) ⊗[R] M) ≃ₗ[R] M ⧸ (I • (⊤ : Submodule R M)) :=
  quotientTensorEquiv M I ≪≫ₗ
  (Submodule.Quotient.equiv _ _ (TensorProduct.lid R M) <| by
    erw [← LinearMap.range_comp, ← (Submodule.topEquiv.lTensor I).range_comp,
      Submodule.smul_eq_map₂, map₂_eq_range_lift_comp_mapIncl]
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      I : Ideal R
      ⊢ Eq (LinearMap.range (((↑(TensorProduct.lid R M)).comp (TensorProduct.map (Su …
    -/
    exact congr_arg _ (TensorProduct.ext' fun _ _ ↦  rfl))
    /-
      🎉 no goals
    -/


variable (M) in
/-- Right tensoring a module with a quotient of the ring is the same as
quotienting that module by the corresponding submodule. -/
noncomputable def tensorQuotEquivQuotSMul (I : Ideal R) :
    (M ⊗[R] (R ⧸ I)) ≃ₗ[R] M ⧸ (I • (⊤ : Submodule R M)) :=
  TensorProduct.comm _ _ _ ≪≫ₗ quotTensorEquivQuotSMul M I


@[simp]
lemma quotTensorEquivQuotSMul_mk_tmul (I : Ideal R) (r : R) (x : M) :
    quotTensorEquivQuotSMul M I (Ideal.Quotient.mk I r ⊗ₜ[R] x) =
      Submodule.Quotient.mk (r • x) :=
  (quotTensorEquivQuotSMul M I).eq_symm_apply.mp <|
    Eq.trans (congrArg (· ⊗ₜ[R] x) <|
        Eq.trans (congrArg (Ideal.Quotient.mk I)
                    (Eq.trans (smul_eq_mul R) (mul_one r))).symm <|
          Submodule.Quotient.mk_smul I r 1) <|
      smul_tmul r _ x


lemma quotTensorEquivQuotSMul_comp_mkQ_rTensor (I : Ideal R) :
    quotTensorEquivQuotSMul M I ∘ₗ I.mkQ.rTensor M =
      (I • ⊤ : Submodule R M).mkQ ∘ₗ TensorProduct.lid R M :=
  TensorProduct.ext' (quotTensorEquivQuotSMul_mk_tmul I)


@[simp]
lemma quotTensorEquivQuotSMul_symm_mk (I : Ideal R) (x : M) :
    (quotTensorEquivQuotSMul M I).symm (Submodule.Quotient.mk x) = 1 ⊗ₜ[R] x :=
  rfl


lemma quotTensorEquivQuotSMul_symm_comp_mkQ (I : Ideal R) :
    (quotTensorEquivQuotSMul M I).symm ∘ₗ (I • ⊤ : Submodule R M).mkQ =
      TensorProduct.mk R (R ⧸ I) M 1 :=
  LinearMap.ext (quotTensorEquivQuotSMul_symm_mk I)


lemma quotTensorEquivQuotSMul_comp_mk (I : Ideal R) :
    quotTensorEquivQuotSMul M I ∘ₗ TensorProduct.mk R (R ⧸ I) M 1 =
      Submodule.mkQ (I • ⊤) :=
  Eq.symm <| (LinearEquiv.toLinearMap_symm_comp_eq _ _).mp <|
    quotTensorEquivQuotSMul_symm_comp_mkQ I


@[simp]
lemma tensorQuotEquivQuotSMul_tmul_mk (I : Ideal R) (x : M) (r : R) :
    tensorQuotEquivQuotSMul M I (x ⊗ₜ[R] Ideal.Quotient.mk I r) =
      Submodule.Quotient.mk (r • x) :=
  quotTensorEquivQuotSMul_mk_tmul I r x


lemma tensorQuotEquivQuotSMul_comp_mkQ_lTensor (I : Ideal R) :
    tensorQuotEquivQuotSMul M I ∘ₗ I.mkQ.lTensor M =
      (I • ⊤ : Submodule R M).mkQ ∘ₗ TensorProduct.rid R M :=
  TensorProduct.ext' (tensorQuotEquivQuotSMul_tmul_mk I)


@[simp]
lemma tensorQuotEquivQuotSMul_symm_mk (I : Ideal R) (x : M) :
    (tensorQuotEquivQuotSMul M I).symm (Submodule.Quotient.mk x) = x ⊗ₜ[R] 1 :=
  rfl


lemma tensorQuotEquivQuotSMul_symm_comp_mkQ (I : Ideal R) :
    (tensorQuotEquivQuotSMul M I).symm ∘ₗ (I • ⊤ : Submodule R M).mkQ =
      (TensorProduct.mk R M (R ⧸ I)).flip 1 :=
  LinearMap.ext (tensorQuotEquivQuotSMul_symm_mk I)


lemma tensorQuotEquivQuotSMul_comp_mk (I : Ideal R) :
    tensorQuotEquivQuotSMul M I ∘ₗ (TensorProduct.mk R M (R ⧸ I)).flip 1 =
      Submodule.mkQ (I • ⊤) :=
  Eq.symm <| (LinearEquiv.toLinearMap_symm_comp_eq _ _).mp <|
    tensorQuotEquivQuotSMul_symm_comp_mkQ I


