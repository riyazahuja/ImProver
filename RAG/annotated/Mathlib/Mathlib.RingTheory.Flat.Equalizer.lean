lemma Module.Flat.ker_lTensor_eq [Module.Flat R M] :
    LinearMap.ker (AlgebraTensorModule.lTensor S M f) =
      LinearMap.range (AlgebraTensorModule.lTensor S M (LinearMap.ker f).subtype) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    M : Type u_3
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : Module S M
    inst✝⁵ : IsScalarTower R S M
    N : Type u_4
    P : Type u_5
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R N
    inst✝¹ : Module R P
    f : LinearMap (RingHom.id R) N P
    inst✝ : Module.Flat R M
    ⊢ Eq (LinearMap.ker ((TensorProduct.AlgebraTensorModule.lTensor S M) f)) (Line …
  -/
  rw [← LinearMap.exact_iff]
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    M : Type u_3
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : Module S M
    inst✝⁵ : IsScalarTower R S M
    N : Type u_4
    P : Type u_5
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R N
    inst✝¹ : Module R P
    f : LinearMap (RingHom.id R) N P
    inst✝ : Module.Flat R M
    ⊢ Function.Exact ⇑((TensorProduct.AlgebraTensorModule.lTensor S M) (LinearMap. …
  -/
  exact Module.Flat.lTensor_exact M (LinearMap.exact_subtype_ker_map f)
  /-
    🎉 no goals
  -/


lemma Module.Flat.eqLocus_lTensor_eq [Module.Flat R M] :
    LinearMap.eqLocus (AlgebraTensorModule.lTensor S M f)
      (AlgebraTensorModule.lTensor S M g) =
      LinearMap.range (AlgebraTensorModule.lTensor S M (LinearMap.eqLocus f g).subtype) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    M : Type u_3
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : Module S M
    inst✝⁵ : IsScalarTower R S M
    N : Type u_4
    P : Type u_5
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R N
    inst✝¹ : Module R P
    f g : LinearMap (RingHom.id R) N P
    inst✝ : Module.Flat R M
    ⊢ Eq (LinearMap.eqLocus ((TensorProduct.AlgebraTensorModule.lTensor S M) f) (( …
  -/
  rw [LinearMap.eqLocus_eq_ker_sub, LinearMap.eqLocus_eq_ker_sub]
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    M : Type u_3
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : Module S M
    inst✝⁵ : IsScalarTower R S M
    N : Type u_4
    P : Type u_5
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R N
    inst✝¹ : Module R P
    f g : LinearMap (RingHom.id R) N P
    inst✝ : Module.Flat R M
    ⊢ Eq (LinearMap.ker (HSub.hSub ((TensorProduct.AlgebraTensorModule.lTensor S M …
  -/
  rw [← map_sub, ker_lTensor_eq]
  /-
    🎉 no goals
  -/


/-- The bilinear map corresponding to `LinearMap.tensorEqLocus`. -/
def LinearMap.tensorEqLocusBil :
    M →ₗ[S] LinearMap.eqLocus f g →ₗ[R]
      LinearMap.eqLocus (AlgebraTensorModule.lTensor S M f)
        (AlgebraTensorModule.lTensor S M g) where
  toFun m :=
                                   /-
                                     R : Type u_1
                                     S : Type u_2
                                     inst✝¹⁰ : CommRing R
                                     inst✝⁹ : CommRing S
                                     inst✝⁸ : Algebra R S
                                     M : Type u_3
                                     inst✝⁷ : AddCommGroup M
                                     inst✝⁶ : Module R M
                                     inst✝⁵ : Module S M
                                     inst✝⁴ : IsScalarTower R S M
                                     N : Type u_4
                                     P : Type u_5
                                     inst✝³ : AddCommGroup N
                                     inst✝² : AddCommGroup P
                                     inst✝¹ : Module R N
                                     inst✝ : Module R P
                                     f g : LinearMap (RingHom.id R) N P
                                     m : M
                                     a : Subtype fun x => Membership.mem (LinearMap.eqLocus f g) x
                                     ⊢ Membership.mem (LinearMap.eqLocus ((TensorProduct.AlgebraTensorModule.lTenso …
                                   -/
    { toFun := fun a ↦ ⟨m ⊗ₜ a, by simp [show f a = g a from a.property]⟩
                                   /-
                                     🎉 no goals
                                   -/
                               /-
                                 R : Type u_1
                                 S : Type u_2
                                 inst✝¹⁰ : CommRing R
                                 inst✝⁹ : CommRing S
                                 inst✝⁸ : Algebra R S
                                 M : Type u_3
                                 inst✝⁷ : AddCommGroup M
                                 inst✝⁶ : Module R M
                                 inst✝⁵ : Module S M
                                 inst✝⁴ : IsScalarTower R S M
                                 N : Type u_4
                                 P : Type u_5
                                 inst✝³ : AddCommGroup N
                                 inst✝² : AddCommGroup P
                                 inst✝¹ : Module R N
                                 inst✝ : Module R P
                                 f g : LinearMap (RingHom.id R) N P
                                 m : M
                                 x y : Subtype fun x => Membership.mem (LinearMap.eqLocus f g) x
                                 ⊢ Eq ((fun a => ⟨TensorProduct.tmul R m ↑a, ⋯⟩) (HAdd.hAdd x y)) (HAdd.hAdd (( …
                               -/
      map_add' := fun x y ↦ by simp [tmul_add]
                               /-
                                 🎉 no goals
                               -/
                                /-
                                  R : Type u_1
                                  S : Type u_2
                                  inst✝¹⁰ : CommRing R
                                  inst✝⁹ : CommRing S
                                  inst✝⁸ : Algebra R S
                                  M : Type u_3
                                  inst✝⁷ : AddCommGroup M
                                  inst✝⁶ : Module R M
                                  inst✝⁵ : Module S M
                                  inst✝⁴ : IsScalarTower R S M
                                  N : Type u_4
                                  P : Type u_5
                                  inst✝³ : AddCommGroup N
                                  inst✝² : AddCommGroup P
                                  inst✝¹ : Module R N
                                  inst✝ : Module R P
                                  f g : LinearMap (RingHom.id R) N P
                                  m : M
                                  r : R
                                  x : Subtype fun x => Membership.mem (LinearMap.eqLocus f g) x
                                  ⊢ Eq ({ toFun := fun a => ⟨TensorProduct.tmul R m ↑a, ⋯⟩, map_add' := ⋯ }.toFu …
                                -/
      map_smul' := fun r x ↦ by simp }
                                /-
                                  🎉 no goals
                                -/
  map_add' x y := by
    /-
      R : Type u_1
      S : Type u_2
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      M : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : Module S M
      inst✝⁴ : IsScalarTower R S M
      N : Type u_4
      P : Type u_5
      inst✝³ : AddCommGroup N
      inst✝² : AddCommGroup P
      inst✝¹ : Module R N
      inst✝ : Module R P
      f g : LinearMap (RingHom.id R) N P
      x y : M
      ⊢ Eq ((fun m => { toFun := fun a => ⟨TensorProduct.tmul R m ↑a, ⋯⟩, map_add' : …
    -/
    ext
    /-
      case h.a
      R : Type u_1
      S : Type u_2
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      M : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : Module S M
      inst✝⁴ : IsScalarTower R S M
      N : Type u_4
      P : Type u_5
      inst✝³ : AddCommGroup N
      inst✝² : AddCommGroup P
      inst✝¹ : Module R N
      inst✝ : Module R P
      f g : LinearMap (RingHom.id R) N P
      x y : M
      x✝ : Subtype fun x => Membership.mem (LinearMap.eqLocus f g) x
      ⊢ Eq ↑(((fun m => { toFun := fun a => ⟨TensorProduct.tmul R m ↑a, ⋯⟩, map_add' …
    -/
    simp [add_tmul]
    /-
      🎉 no goals
    -/
  map_smul' r x := by
    /-
      R : Type u_1
      S : Type u_2
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      M : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : Module S M
      inst✝⁴ : IsScalarTower R S M
      N : Type u_4
      P : Type u_5
      inst✝³ : AddCommGroup N
      inst✝² : AddCommGroup P
      inst✝¹ : Module R N
      inst✝ : Module R P
      f g : LinearMap (RingHom.id R) N P
      r : S
      x : M
      ⊢ Eq ({ toFun := fun m => { toFun := fun a => ⟨TensorProduct.tmul R m ↑a, ⋯⟩,  …
    -/
    ext
    /-
      case h.a
      R : Type u_1
      S : Type u_2
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      M : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : Module S M
      inst✝⁴ : IsScalarTower R S M
      N : Type u_4
      P : Type u_5
      inst✝³ : AddCommGroup N
      inst✝² : AddCommGroup P
      inst✝¹ : Module R N
      inst✝ : Module R P
      f g : LinearMap (RingHom.id R) N P
      r : S
      x : M
      x✝ : Subtype fun x => Membership.mem (LinearMap.eqLocus f g) x
      ⊢ Eq ↑(({ toFun := fun m => { toFun := fun a => ⟨TensorProduct.tmul R m ↑a, ⋯⟩ …
    -/
    simp [smul_tmul']
    /-
      🎉 no goals
    -/


/-- The bilinear map corresponding to `LinearMap.tensorKer`. -/
def LinearMap.tensorKerBil :
    M →ₗ[S] LinearMap.ker f →ₗ[R] LinearMap.ker (AlgebraTensorModule.lTensor S M f) where
  toFun m :=
                                   /-
                                     R : Type u_1
                                     S : Type u_2
                                     inst✝¹⁰ : CommRing R
                                     inst✝⁹ : CommRing S
                                     inst✝⁸ : Algebra R S
                                     M : Type u_3
                                     inst✝⁷ : AddCommGroup M
                                     inst✝⁶ : Module R M
                                     inst✝⁵ : Module S M
                                     inst✝⁴ : IsScalarTower R S M
                                     N : Type u_4
                                     P : Type u_5
                                     inst✝³ : AddCommGroup N
                                     inst✝² : AddCommGroup P
                                     inst✝¹ : Module R N
                                     inst✝ : Module R P
                                     f g : LinearMap (RingHom.id R) N P
                                     m : M
                                     a : Subtype fun x => Membership.mem (LinearMap.ker f) x
                                     ⊢ Membership.mem (LinearMap.ker ((TensorProduct.AlgebraTensorModule.lTensor S  …
                                   -/
    { toFun := fun a ↦ ⟨m ⊗ₜ a, by simp⟩
                                   /-
                                     🎉 no goals
                                   -/
                               /-
                                 R : Type u_1
                                 S : Type u_2
                                 inst✝¹⁰ : CommRing R
                                 inst✝⁹ : CommRing S
                                 inst✝⁸ : Algebra R S
                                 M : Type u_3
                                 inst✝⁷ : AddCommGroup M
                                 inst✝⁶ : Module R M
                                 inst✝⁵ : Module S M
                                 inst✝⁴ : IsScalarTower R S M
                                 N : Type u_4
                                 P : Type u_5
                                 inst✝³ : AddCommGroup N
                                 inst✝² : AddCommGroup P
                                 inst✝¹ : Module R N
                                 inst✝ : Module R P
                                 f g : LinearMap (RingHom.id R) N P
                                 m : M
                                 x y : Subtype fun x => Membership.mem (LinearMap.ker f) x
                                 ⊢ Eq ((fun a => ⟨TensorProduct.tmul R m ↑a, ⋯⟩) (HAdd.hAdd x y)) (HAdd.hAdd (( …
                               -/
      map_add' := fun x y ↦ by simp [tmul_add]
                               /-
                                 🎉 no goals
                               -/
                                /-
                                  R : Type u_1
                                  S : Type u_2
                                  inst✝¹⁰ : CommRing R
                                  inst✝⁹ : CommRing S
                                  inst✝⁸ : Algebra R S
                                  M : Type u_3
                                  inst✝⁷ : AddCommGroup M
                                  inst✝⁶ : Module R M
                                  inst✝⁵ : Module S M
                                  inst✝⁴ : IsScalarTower R S M
                                  N : Type u_4
                                  P : Type u_5
                                  inst✝³ : AddCommGroup N
                                  inst✝² : AddCommGroup P
                                  inst✝¹ : Module R N
                                  inst✝ : Module R P
                                  f g : LinearMap (RingHom.id R) N P
                                  m : M
                                  r : R
                                  x : Subtype fun x => Membership.mem (LinearMap.ker f) x
                                  ⊢ Eq ({ toFun := fun a => ⟨TensorProduct.tmul R m ↑a, ⋯⟩, map_add' := ⋯ }.toFu …
                                -/
      map_smul' := fun r x ↦ by simp }
                                /-
                                  🎉 no goals
                                -/
                     /-
                       R : Type u_1
                       S : Type u_2
                       inst✝¹⁰ : CommRing R
                       inst✝⁹ : CommRing S
                       inst✝⁸ : Algebra R S
                       M : Type u_3
                       inst✝⁷ : AddCommGroup M
                       inst✝⁶ : Module R M
                       inst✝⁵ : Module S M
                       inst✝⁴ : IsScalarTower R S M
                       N : Type u_4
                       P : Type u_5
                       inst✝³ : AddCommGroup N
                       inst✝² : AddCommGroup P
                       inst✝¹ : Module R N
                       inst✝ : Module R P
                       f g : LinearMap (RingHom.id R) N P
                       x y : M
                       ⊢ Eq ((fun m => { toFun := fun a => ⟨TensorProduct.tmul R m ↑a, ⋯⟩, map_add' : …
                     -/
  map_add' x y := by ext; simp [add_tmul]
                          /-
                            🎉 no goals
                          -/
                      /-
                        R : Type u_1
                        S : Type u_2
                        inst✝¹⁰ : CommRing R
                        inst✝⁹ : CommRing S
                        inst✝⁸ : Algebra R S
                        M : Type u_3
                        inst✝⁷ : AddCommGroup M
                        inst✝⁶ : Module R M
                        inst✝⁵ : Module S M
                        inst✝⁴ : IsScalarTower R S M
                        N : Type u_4
                        P : Type u_5
                        inst✝³ : AddCommGroup N
                        inst✝² : AddCommGroup P
                        inst✝¹ : Module R N
                        inst✝ : Module R P
                        f g : LinearMap (RingHom.id R) N P
                        r : S
                        x : M
                        ⊢ Eq ({ toFun := fun m => { toFun := fun a => ⟨TensorProduct.tmul R m ↑a, ⋯⟩,  …
                      -/
  map_smul' r x := by ext y; simp [smul_tmul']
                             /-
                               🎉 no goals
                             -/


/-- The canonical map `M ⊗[R] eq(f, g) →ₗ[R] eq(𝟙 ⊗ f, 𝟙 ⊗ g)`. -/
def LinearMap.tensorEqLocus : M ⊗[R] (LinearMap.eqLocus f g) →ₗ[S]
    LinearMap.eqLocus (AlgebraTensorModule.lTensor S M f) (AlgebraTensorModule.lTensor S M g) :=
  AlgebraTensorModule.lift (tensorEqLocusBil S M f g)


/-- The canonical map `M ⊗[R] ker f →ₗ[R] ker (𝟙 ⊗ f)`. -/
def LinearMap.tensorKer : M ⊗[R] (LinearMap.ker f) →ₗ[S]
    LinearMap.ker (AlgebraTensorModule.lTensor S M f) :=
  AlgebraTensorModule.lift (f.tensorKerBil S M)


@[simp]
lemma LinearMap.tensorKer_tmul (m : M) (x : LinearMap.ker f) :
    (tensorKer S M f (m ⊗ₜ[R] x) : M ⊗[R] N) = m ⊗ₜ[R] (x : N) :=
  rfl


@[simp]
lemma LinearMap.tensorKer_coe (x : M ⊗[R] (LinearMap.ker f)) :
    (tensorKer S M f x : M ⊗[R] N) = (ker f).subtype.lTensor M x := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    M : Type u_3
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : Module S M
    inst✝⁴ : IsScalarTower R S M
    N : Type u_4
    P : Type u_5
    inst✝³ : AddCommGroup N
    inst✝² : AddCommGroup P
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) N P
    x : TensorProduct R M (Subtype fun x => Membership.mem (LinearMap.ker f) x)
    ⊢ Eq (↑((LinearMap.tensorKer S M f) x)) ((LinearMap.lTensor M (LinearMap.ker f …
  -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
  induction x <;> simp_all
                  /-
                    🎉 no goals
                  -/


@[simp]
lemma LinearMap.tensorEqLocus_tmul (m : M) (x : LinearMap.eqLocus f g) :
    (tensorEqLocus S M f g (m ⊗ₜ[R] x) : M ⊗[R] N) = m ⊗ₜ[R] (x : N) :=
  rfl


@[simp]
lemma LinearMap.tensorEqLocus_coe (x : M ⊗[R] (LinearMap.eqLocus f g)) :
    (tensorEqLocus S M f g x : M ⊗[R] N) = (eqLocus f g).subtype.lTensor M x := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    M : Type u_3
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : Module S M
    inst✝⁴ : IsScalarTower R S M
    N : Type u_4
    P : Type u_5
    inst✝³ : AddCommGroup N
    inst✝² : AddCommGroup P
    inst✝¹ : Module R N
    inst✝ : Module R P
    f g : LinearMap (RingHom.id R) N P
    x : TensorProduct R M (Subtype fun x => Membership.mem (LinearMap.eqLocus f g) …
    ⊢ Eq (↑((LinearMap.tensorEqLocus S M f g) x)) ((LinearMap.lTensor M (LinearMap …
  -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
  induction x <;> simp_all
                  /-
                    🎉 no goals
                  -/


private def LinearMap.tensorKerInv [Module.Flat R M] :
    ker (AlgebraTensorModule.lTensor S M f) →ₗ[S] M ⊗[R] (ker f) :=
  LinearMap.codRestrictOfInjective (LinearMap.ker (AlgebraTensorModule.lTensor S M f)).subtype
    (AlgebraTensorModule.lTensor S M (ker f).subtype)
    (Module.Flat.lTensor_preserves_injective_linearMap (ker f).subtype
                                     /-
                                       R : Type u_1
                                       S : Type u_2
                                       inst✝¹¹ : CommRing R
                                       inst✝¹⁰ : CommRing S
                                       inst✝⁹ : Algebra R S
                                       M : Type u_3
                                       inst✝⁸ : AddCommGroup M
                                       inst✝⁷ : Module R M
                                       inst✝⁶ : Module S M
                                       inst✝⁵ : IsScalarTower R S M
                                       N : Type u_4
                                       P : Type u_5
                                       inst✝⁴ : AddCommGroup N
                                       inst✝³ : AddCommGroup P
                                       inst✝² : Module R N
                                       inst✝¹ : Module R P
                                       f g : LinearMap (RingHom.id R) N P
                                       inst✝ : Module.Flat R M
                                       ⊢ ∀ (x : Subtype fun x => Membership.mem (LinearMap.ker ((TensorProduct.Algebr …
                                     -/
      (ker f).injective_subtype) (by simp [Module.Flat.ker_lTensor_eq])
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
private lemma LinearMap.lTensor_ker_subtype_tensorKerInv [Module.Flat R M]
    (x : ker (AlgebraTensorModule.lTensor S M f)) :
    (lTensor M (ker f).subtype) ((tensorKerInv S M f) x) = x := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    M : Type u_3
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : Module S M
    inst✝⁵ : IsScalarTower R S M
    N : Type u_4
    P : Type u_5
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R N
    inst✝¹ : Module R P
    f : LinearMap (RingHom.id R) N P
    inst✝ : Module.Flat R M
    x : Subtype fun x => Membership.mem (LinearMap.ker ((TensorProduct.AlgebraTens …
    ⊢ Eq ((LinearMap.lTensor M (LinearMap.ker f).subtype) ((LinearMap.tensorKerInv …
  -/
  rw [← AlgebraTensorModule.coe_lTensor (A := S)]
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    M : Type u_3
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : Module S M
    inst✝⁵ : IsScalarTower R S M
    N : Type u_4
    P : Type u_5
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R N
    inst✝¹ : Module R P
    f : LinearMap (RingHom.id R) N P
    inst✝ : Module.Flat R M
    x : Subtype fun x => Membership.mem (LinearMap.ker ((TensorProduct.AlgebraTens …
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.lTensor S M) (LinearMap.ker f).subty …
  -/
  simp [LinearMap.tensorKerInv]
  /-
    🎉 no goals
  -/


private def LinearMap.tensorEqLocusInv [Module.Flat R M] :
    eqLocus (AlgebraTensorModule.lTensor S M f) (AlgebraTensorModule.lTensor S M g) →ₗ[S]
      M ⊗[R] (eqLocus f g) :=
  LinearMap.codRestrictOfInjective
    (LinearMap.eqLocus (AlgebraTensorModule.lTensor S M f)
      (AlgebraTensorModule.lTensor S M g)).subtype
    (AlgebraTensorModule.lTensor S M (eqLocus f g).subtype)
    (Module.Flat.lTensor_preserves_injective_linearMap (eqLocus f g).subtype
                                           /-
                                             R : Type u_1
                                             S : Type u_2
                                             inst✝¹¹ : CommRing R
                                             inst✝¹⁰ : CommRing S
                                             inst✝⁹ : Algebra R S
                                             M : Type u_3
                                             inst✝⁸ : AddCommGroup M
                                             inst✝⁷ : Module R M
                                             inst✝⁶ : Module S M
                                             inst✝⁵ : IsScalarTower R S M
                                             N : Type u_4
                                             P : Type u_5
                                             inst✝⁴ : AddCommGroup N
                                             inst✝³ : AddCommGroup P
                                             inst✝² : Module R N
                                             inst✝¹ : Module R P
                                             f g : LinearMap (RingHom.id R) N P
                                             inst✝ : Module.Flat R M
                                             ⊢ ∀ (x : Subtype fun x => Membership.mem (LinearMap.eqLocus ((TensorProduct.Al …
                                           -/
      (eqLocus f g).injective_subtype) (by simp [Module.Flat.eqLocus_lTensor_eq])
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
private lemma LinearMap.lTensor_eqLocus_subtype_tensorEqLocusInv [Module.Flat R M]
    (x : eqLocus (AlgebraTensorModule.lTensor S M f) (AlgebraTensorModule.lTensor S M g)) :
    (lTensor M (eqLocus f g).subtype) (tensorEqLocusInv S M f g x) = x := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    M : Type u_3
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : Module S M
    inst✝⁵ : IsScalarTower R S M
    N : Type u_4
    P : Type u_5
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R N
    inst✝¹ : Module R P
    f g : LinearMap (RingHom.id R) N P
    inst✝ : Module.Flat R M
    x : Subtype fun x => Membership.mem (LinearMap.eqLocus ((TensorProduct.Algebra …
    ⊢ Eq ((LinearMap.lTensor M (LinearMap.eqLocus f g).subtype) ((LinearMap.tensor …
  -/
  rw [← AlgebraTensorModule.coe_lTensor (A := S)]
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    M : Type u_3
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : Module S M
    inst✝⁵ : IsScalarTower R S M
    N : Type u_4
    P : Type u_5
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R N
    inst✝¹ : Module R P
    f g : LinearMap (RingHom.id R) N P
    inst✝ : Module.Flat R M
    x : Subtype fun x => Membership.mem (LinearMap.eqLocus ((TensorProduct.Algebra …
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.lTensor S M) (LinearMap.eqLocus f g) …
  -/
  simp [LinearMap.tensorEqLocusInv]
  /-
    🎉 no goals
  -/


/-- If `M` is `R`-flat, the canonical map `M ⊗[R] ker f →ₗ[R] ker (𝟙 ⊗ f)` is an isomorphism. -/
def LinearMap.tensorKerEquiv [Module.Flat R M] :
    M ⊗[R] LinearMap.ker f ≃ₗ[S] LinearMap.ker (AlgebraTensorModule.lTensor S M f) :=
  LinearEquiv.ofLinear (LinearMap.tensorKer S M f) (LinearMap.tensorKerInv S M f)
        /-
          R : Type u_1
          S : Type u_2
          inst✝¹¹ : CommRing R
          inst✝¹⁰ : CommRing S
          inst✝⁹ : Algebra R S
          M : Type u_3
          inst✝⁸ : AddCommGroup M
          inst✝⁷ : Module R M
          inst✝⁶ : Module S M
          inst✝⁵ : IsScalarTower R S M
          N : Type u_4
          P : Type u_5
          inst✝⁴ : AddCommGroup N
          inst✝³ : AddCommGroup P
          inst✝² : Module R N
          inst✝¹ : Module R P
          f g : LinearMap (RingHom.id R) N P
          inst✝ : Module.Flat R M
          ⊢ Eq ((LinearMap.tensorKer S M f).comp (LinearMap.tensorKerInv S M f)) LinearM …
        -/
    (by ext x; simp)
               /-
                 🎉 no goals
               -/
    (by
      /-
        R : Type u_1
        S : Type u_2
        inst✝¹¹ : CommRing R
        inst✝¹⁰ : CommRing S
        inst✝⁹ : Algebra R S
        M : Type u_3
        inst✝⁸ : AddCommGroup M
        inst✝⁷ : Module R M
        inst✝⁶ : Module S M
        inst✝⁵ : IsScalarTower R S M
        N : Type u_4
        P : Type u_5
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R N
        inst✝¹ : Module R P
        f g : LinearMap (RingHom.id R) N P
        inst✝ : Module.Flat R M
        ⊢ Eq ((LinearMap.tensorKerInv S M f).comp (LinearMap.tensorKer S M f)) LinearM …
      -/
      ext m x
      apply (Module.Flat.lTensor_preserves_injective_linearMap (ker f).subtype
        (ker f).injective_subtype)
      /-
        case a.h.h.a
        R : Type u_1
        S : Type u_2
        inst✝¹¹ : CommRing R
        inst✝¹⁰ : CommRing S
        inst✝⁹ : Algebra R S
        M : Type u_3
        inst✝⁸ : AddCommGroup M
        inst✝⁷ : Module R M
        inst✝⁶ : Module S M
        inst✝⁵ : IsScalarTower R S M
        N : Type u_4
        P : Type u_5
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R N
        inst✝¹ : Module R P
        f g : LinearMap (RingHom.id R) N P
        inst✝ : Module.Flat R M
        m : M
        x : Subtype fun x => Membership.mem (LinearMap.ker f) x
        ⊢ Eq ((LinearMap.lTensor M (LinearMap.ker f).subtype) (((TensorProduct.Algebra …
      -/
      simp)
      /-
        🎉 no goals
      -/


@[simp]
lemma LinearMap.tensorKerEquiv_apply [Module.Flat R M] (x : M ⊗[R] ker f) :
    tensorKerEquiv S M f x = tensorKer S M f x :=
  rfl


@[simp]
lemma LinearMap.lTensor_ker_subtype_tensorKerEquiv_symm [Module.Flat R M]
    (x : ker (AlgebraTensorModule.lTensor S M f)) :
    (lTensor M (ker f).subtype) ((tensorKerEquiv S M f).symm x) = x :=
  lTensor_ker_subtype_tensorKerInv S M f x


/-- If `M` is `R`-flat, the canonical map `M ⊗[R] eq(f, g) →ₗ[S] eq (𝟙 ⊗ f, 𝟙 ⊗ g)` is an
isomorphism. -/
def LinearMap.tensorEqLocusEquiv [Module.Flat R M] :
    M ⊗[R] eqLocus f g ≃ₗ[S]
      eqLocus (AlgebraTensorModule.lTensor S M f)
        (AlgebraTensorModule.lTensor S M g) :=
  LinearEquiv.ofLinear (LinearMap.tensorEqLocus S M f g) (LinearMap.tensorEqLocusInv S M f g)
        /-
          R : Type u_1
          S : Type u_2
          inst✝¹¹ : CommRing R
          inst✝¹⁰ : CommRing S
          inst✝⁹ : Algebra R S
          M : Type u_3
          inst✝⁸ : AddCommGroup M
          inst✝⁷ : Module R M
          inst✝⁶ : Module S M
          inst✝⁵ : IsScalarTower R S M
          N : Type u_4
          P : Type u_5
          inst✝⁴ : AddCommGroup N
          inst✝³ : AddCommGroup P
          inst✝² : Module R N
          inst✝¹ : Module R P
          f g : LinearMap (RingHom.id R) N P
          inst✝ : Module.Flat R M
          ⊢ Eq ((LinearMap.tensorEqLocus S M f g).comp (LinearMap.tensorEqLocusInv S M f …
        -/
    (by ext; simp)
             /-
               🎉 no goals
             -/
    (by
      /-
        R : Type u_1
        S : Type u_2
        inst✝¹¹ : CommRing R
        inst✝¹⁰ : CommRing S
        inst✝⁹ : Algebra R S
        M : Type u_3
        inst✝⁸ : AddCommGroup M
        inst✝⁷ : Module R M
        inst✝⁶ : Module S M
        inst✝⁵ : IsScalarTower R S M
        N : Type u_4
        P : Type u_5
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R N
        inst✝¹ : Module R P
        f g : LinearMap (RingHom.id R) N P
        inst✝ : Module.Flat R M
        ⊢ Eq ((LinearMap.tensorEqLocusInv S M f g).comp (LinearMap.tensorEqLocus S M f …
      -/
      ext m x
      apply (Module.Flat.lTensor_preserves_injective_linearMap (eqLocus f g).subtype
        (eqLocus f g).injective_subtype)
      /-
        case a.h.h.a
        R : Type u_1
        S : Type u_2
        inst✝¹¹ : CommRing R
        inst✝¹⁰ : CommRing S
        inst✝⁹ : Algebra R S
        M : Type u_3
        inst✝⁸ : AddCommGroup M
        inst✝⁷ : Module R M
        inst✝⁶ : Module S M
        inst✝⁵ : IsScalarTower R S M
        N : Type u_4
        P : Type u_5
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R N
        inst✝¹ : Module R P
        f g : LinearMap (RingHom.id R) N P
        inst✝ : Module.Flat R M
        m : M
        x : Subtype fun x => Membership.mem (LinearMap.eqLocus f g) x
        ⊢ Eq ((LinearMap.lTensor M (LinearMap.eqLocus f g).subtype) (((TensorProduct.A …
      -/
      simp)
      /-
        🎉 no goals
      -/


@[simp]
lemma LinearMap.tensorEqLocusEquiv_apply [Module.Flat R M] (x : M ⊗[R] LinearMap.eqLocus f g) :
    LinearMap.tensorEqLocusEquiv S M f g x = LinearMap.tensorEqLocus S M f g x :=
  rfl


@[simp]
lemma LinearMap.lTensor_eqLocus_subtype_tensoreqLocusEquiv_symm [Module.Flat R M]
    (x : eqLocus (AlgebraTensorModule.lTensor S M f) (AlgebraTensorModule.lTensor S M g)) :
    (lTensor M (eqLocus f g).subtype) ((tensorEqLocusEquiv S M f g).symm x) = x :=
  lTensor_eqLocus_subtype_tensorEqLocusInv S M f g x


private def AlgHom.tensorEqualizerAux :
    T ⊗[R] AlgHom.equalizer f g →ₗ[S]
      AlgHom.equalizer (Algebra.TensorProduct.map (AlgHom.id S T) f)
        (Algebra.TensorProduct.map (AlgHom.id S T) g) :=
  LinearMap.tensorEqLocus S T (f : A →ₗ[R] B) (g : A →ₗ[R] B)


private local instance : AddHomClass (A →ₐ[R] B) A B := inferInstance


@[simp]
private lemma AlgHom.coe_tensorEqualizerAux (x : T ⊗[R] AlgHom.equalizer f g) :
    (AlgHom.tensorEqualizerAux S T f g x : T ⊗[R] A) =
      Algebra.TensorProduct.map (AlgHom.id S T) (AlgHom.equalizer f g).val x := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    T : Type u_3
    inst✝⁷ : CommRing T
    inst✝⁶ : Algebra R T
    inst✝⁵ : Algebra S T
    inst✝⁴ : IsScalarTower R S T
    A : Type u_4
    B : Type u_5
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    f g : AlgHom R A B
    x : TensorProduct R T (Subtype fun x => Membership.mem (AlgHom.equalizer f g) x)
    ⊢ Eq (↑((AlgHom.tensorEqualizerAux S T f g) x)) ((Algebra.TensorProduct.map (A …
  -/
  induction' x with x y x y hx hy
    /-
      case zero
      R : Type u_1
      S : Type u_2
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      T : Type u_3
      inst✝⁷ : CommRing T
      inst✝⁶ : Algebra R T
      inst✝⁵ : Algebra S T
      inst✝⁴ : IsScalarTower R S T
      A : Type u_4
      B : Type u_5
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      f g : AlgHom R A B
      ⊢ Eq (↑((AlgHom.tensorEqualizerAux S T f g) 0)) ((Algebra.TensorProduct.map (A …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case tmul
      R : Type u_1
      S : Type u_2
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      T : Type u_3
      inst✝⁷ : CommRing T
      inst✝⁶ : Algebra R T
      inst✝⁵ : Algebra S T
      inst✝⁴ : IsScalarTower R S T
      A : Type u_4
      B : Type u_5
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      f g : AlgHom R A B
      x : T
      y : Subtype fun x => Membership.mem (AlgHom.equalizer f g) x
      ⊢ Eq (↑((AlgHom.tensorEqualizerAux S T f g) (TensorProduct.tmul R x y))) ((Alg …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case add
      R : Type u_1
      S : Type u_2
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      T : Type u_3
      inst✝⁷ : CommRing T
      inst✝⁶ : Algebra R T
      inst✝⁵ : Algebra S T
      inst✝⁴ : IsScalarTower R S T
      A : Type u_4
      B : Type u_5
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      f g : AlgHom R A B
      x y : TensorProduct R T (Subtype fun x => Membership.mem (AlgHom.equalizer f g …
      hx : Eq (↑((AlgHom.tensorEqualizerAux S T f g) x)) ((Algebra.TensorProduct.map …
      hy : Eq (↑((AlgHom.tensorEqualizerAux S T f g) y)) ((Algebra.TensorProduct.map …
      ⊢ Eq (↑((AlgHom.tensorEqualizerAux S T f g) (HAdd.hAdd x y))) ((Algebra.Tensor …
    -/
  · simp [hx, hy]
    /-
      🎉 no goals
    -/


private lemma AlgHom.tensorEqualizerAux_mul (x y : T ⊗[R] AlgHom.equalizer f g) :
    AlgHom.tensorEqualizerAux S T f g (x * y) =
      AlgHom.tensorEqualizerAux S T f g x *
        AlgHom.tensorEqualizerAux S T f g y := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    T : Type u_3
    inst✝⁷ : CommRing T
    inst✝⁶ : Algebra R T
    inst✝⁵ : Algebra S T
    inst✝⁴ : IsScalarTower R S T
    A : Type u_4
    B : Type u_5
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    f g : AlgHom R A B
    x y : TensorProduct R T (Subtype fun x => Membership.mem (AlgHom.equalizer f g …
    ⊢ Eq ((AlgHom.tensorEqualizerAux S T f g) (HMul.hMul x y)) (HMul.hMul ((AlgHom …
  -/
  apply Subtype.ext
  /-
    case a
    R : Type u_1
    S : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    T : Type u_3
    inst✝⁷ : CommRing T
    inst✝⁶ : Algebra R T
    inst✝⁵ : Algebra S T
    inst✝⁴ : IsScalarTower R S T
    A : Type u_4
    B : Type u_5
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    f g : AlgHom R A B
    x y : TensorProduct R T (Subtype fun x => Membership.mem (AlgHom.equalizer f g …
    ⊢ Eq ↑((AlgHom.tensorEqualizerAux S T f g) (HMul.hMul x y)) ↑(HMul.hMul ((AlgH …
  -/
  rw [AlgHom.coe_tensorEqualizerAux]
  /-
    case a
    R : Type u_1
    S : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    T : Type u_3
    inst✝⁷ : CommRing T
    inst✝⁶ : Algebra R T
    inst✝⁵ : Algebra S T
    inst✝⁴ : IsScalarTower R S T
    A : Type u_4
    B : Type u_5
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    f g : AlgHom R A B
    x y : TensorProduct R T (Subtype fun x => Membership.mem (AlgHom.equalizer f g …
    ⊢ Eq ((Algebra.TensorProduct.map (AlgHom.id S T) (AlgHom.equalizer f g).val) ( …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The canonical map `T ⊗[R] eq(f, g) →ₐ[S] eq (𝟙 ⊗ f, 𝟙 ⊗ g)`. -/
def AlgHom.tensorEqualizer :
    T ⊗[R] AlgHom.equalizer f g →ₐ[S]
      AlgHom.equalizer (Algebra.TensorProduct.map (AlgHom.id S T) f)
        (Algebra.TensorProduct.map (AlgHom.id S T) g) :=
  AlgHom.ofLinearMap (AlgHom.tensorEqualizerAux S T f g)
    rfl (AlgHom.tensorEqualizerAux_mul S T f g)


@[simp]
lemma AlgHom.coe_tensorEqualizer (x : T ⊗[R] AlgHom.equalizer f g) :
    (AlgHom.tensorEqualizer S T f g x : T ⊗[R] A) =
      Algebra.TensorProduct.map (AlgHom.id S T) (AlgHom.equalizer f g).val x :=
  AlgHom.coe_tensorEqualizerAux S T f g x


/-- If `T` is `R`-flat, the canonical map
`T ⊗[R] eq(f, g) →ₐ[S] eq (𝟙 ⊗ f, 𝟙 ⊗ g)` is an isomorphism. -/
def AlgHom.tensorEqualizerEquiv [Module.Flat R T] :
    T ⊗[R] AlgHom.equalizer f g ≃ₐ[S]
      AlgHom.equalizer (Algebra.TensorProduct.map (AlgHom.id S T) f)
        (Algebra.TensorProduct.map (AlgHom.id S T) g) :=
  AlgEquiv.ofLinearEquiv (LinearMap.tensorEqLocusEquiv S T f.toLinearMap g.toLinearMap)
    rfl (AlgHom.tensorEqualizerAux_mul S T f g)


@[simp]
lemma AlgHom.tensorEqualizerEquiv_apply [Module.Flat R T]
    (x : T ⊗[R] AlgHom.equalizer f g) :
    AlgHom.tensorEqualizerEquiv S T f g x = AlgHom.tensorEqualizer S T f g x :=
  rfl


