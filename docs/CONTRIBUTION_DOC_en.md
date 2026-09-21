# Contribute Documentation

Welcome to contribute to the PyPTO documentation. Documentation that meets the requirements appears in the [PyPTO Documentation Center](https://pypto.gitcode.com).

This project supports content contribution in Markdown format. You can create documents with the `.md` extension or modify existing documents.

## Writing Requirements

### Basic Specifications

Strictly follow the [Documentation Writing Requirements](https://gitcode.com/cann/community/blob/master/contributor/docs/document_writing_specs.md).

### Terminology Standard Expression Specifications

All domain-specific terms must strictly follow the standard expressions in the [Glossary](./zh/guide/appendix/glossary.md). If a term is not covered by the glossary, you can submit a PR to extend the glossary.

### Product Model Standard Expression

To maintain consistent product model expressions in the documentation system, use the following standard naming conventions:

- Ascend 950PR&950DT series products
- Atlas A3 series products
- Atlas A2 series products

### Product Difference Description Specification

For functional differences between product models, it is recommended to consolidate them into the **Constraint Description** of the corresponding `.md` file (for example, [pypto.bitwise_and](./zh/api/tensor_api/operation/pypto-bitwise_and.md)). This helps users quickly locate difference information and facilitates subsequent documentation customization, extension, and version iteration maintenance.

## Update or Add Documentation

### Update Documentation

If you need to update existing documentation, click the edit button in the upper right corner of the [PyPTO Documentation Center](https://pypto.gitcode.com) page to navigate to the source file. Modify the file and submit a PR to participate in the contribution.

### Add Documentation

If you need to add documentation, create a new Markdown file in the appropriate directory under docs. For the directory structure description, refer to [Directory Description](./README_en.md#directory-description).

1. Create a new file.

2. Edit the directory index file to add the new file to the webpage.

    For API documentation as a sample, when adding a new interface, first find the [index.md](./zh/api/index.md) file in the `docs/zh/api` directory. This file corresponds to the organizational structure of the API documentation.

    Add the new file to the corresponding category, or create a new category and then add it. For example:

    ```{toctree}
    :maxdepth: 2

    tensor/index
    element/index
    operation/index
    datatype/index
    controlflow/index
    config/index
    symbolic/index
    newcategory/index
    ```

    Add the following content to `newcategory/index.md`:

    ```{toctree}
    :maxdepth: 2

    pypto-newapi1
    pypto-newapi1
    ```

After completing the above operations, submit a PR to participate in the contribution.

## Local Build

After submitting a PR, you can use the Sphinx tool to automatically generate documentation locally. For details, refer to [Documentation Build](./README_en.md#documentation-build).

## Remote Build

After the PR is merged, the webpage content is automatically updated. You can then view the added content in the [PyPTO Documentation Center](https://pypto.gitcode.com).
